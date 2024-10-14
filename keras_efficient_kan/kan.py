import numpy as np
import keras
from keras import ops
from keras import backend
from keras.src import initializers
from keras.src.layers import Layer, Dropout, LayerNormalization

@keras.utils.register_keras_serializable(package="keras_efficient_kan", name="GridInitializer")
class GridInitializer(initializers.Initializer):
    def __init__(self, grid_range, grid_size, spline_order):
        self.grid_range = grid_range
        self.grid_size = grid_size
        self.spline_order = spline_order

    def __call__(self, shape, dtype=None):
        h = (self.grid_range[1] - self.grid_range[0]) / self.grid_size
        start = -self.spline_order * h + self.grid_range[0]
        stop = (self.grid_size + self.spline_order) * h + self.grid_range[0]
        num = self.grid_size + 2 * self.spline_order + 1
        
        # Create the grid using numpy
        grid = np.linspace(start, stop, num, dtype=np.float32)
        
        # Repeat the grid for each feature
        grid = np.tile(grid, (shape[1], 1))
        
        # Add the batch dimension
        grid = np.expand_dims(grid, 0)
        
        # Convert to the appropriate backend tensor
        return ops.convert_to_tensor(grid, dtype=dtype)

    def get_config(self):
        return {
            "grid_range": self.grid_range,
            "grid_size": self.grid_size,
            "spline_order": self.spline_order
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@keras.utils.register_keras_serializable(package="keras_efficient_kan", name="KANLinear")
class KANLinear(Layer):
    def __init__(
        self,
        units,
        grid_size=3,
        spline_order=3,
        base_activation='relu',
        grid_range=[-1, 1],
        dropout=0.,
        use_bias=True,
        use_layernorm=True,
        **kwargs
    ):
        super(KANLinear, self).__init__(**kwargs)
        self.units = units
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation_name = base_activation
        self.grid_range = grid_range
        self.use_bias = use_bias
        self.use_layernorm = use_layernorm
        self.dropout_rate = dropout
        self.dropout = Dropout(self.dropout_rate)
        if self.use_layernorm:
            self.layer_norm = LayerNormalization(axis=-1)
        else:
            self.layer_norm = None
        self.in_features = None

    def build(self, input_shape):
        self.in_features = input_shape[-1]
        dtype = backend.floatx()
        
        self.grid = self.add_weight(
            name="grid",
            shape=[1, self.in_features, self.grid_size + 2 * self.spline_order + 1],
            initializer=GridInitializer(self.grid_range, self.grid_size, self.spline_order),
            trainable=False,
            dtype=dtype
        )

        self.base_weight = self.add_weight(
            name="base_weight",
            shape=[self.in_features, self.units],
            initializer='glorot_uniform',
            dtype=dtype
        )
        if self.use_bias:
            self.base_bias = self.add_weight(
                name="base_bias",
                shape=[self.units],
                initializer="zeros",
                dtype=dtype
            )
        self.spline_weight = self.add_weight(
            name="spline_weight",
            shape=[self.in_features * (self.grid_size + self.spline_order), self.units],
            initializer='glorot_uniform',
            dtype=dtype
        )
        if self.use_layernorm:
            self.layer_norm.build(input_shape)
        
        self.built = True

    def call(self, x, training=None):
        input_shape = ops.shape(x)
        x = ops.cast(x, self.dtype)
        x_2d = ops.reshape(x, [-1, self.in_features])
        
        if self.use_layernorm:
            x_2d = self.layer_norm(x_2d)
        
        base_activation = getattr(ops, self.base_activation_name)
        base_output = ops.matmul(base_activation(x_2d), self.base_weight)
        if self.use_bias:
            base_output = ops.add(base_output, self.base_bias)
        
        spline_output = ops.matmul(self.b_splines(x_2d), self.spline_weight)
        output_2d = self.dropout(base_output, training=training) + self.dropout(spline_output, training=training)
        
        # Use ops.reshape with a tuple of integers for the new shape
        new_shape = tuple(input_shape[:-1]) + (self.units,)
        return ops.reshape(output_2d, new_shape)

    def b_splines(self, x):
        x_expanded = ops.expand_dims(x, -1)
        bases = ops.cast((x_expanded >= self.grid[..., :-1]) & (x_expanded < self.grid[..., 1:]), self.dtype)
        
        for k in range(1, self.spline_order + 1):
            left_denominator = self.grid[..., k:-1] - self.grid[..., :-(k + 1)]
            right_denominator = self.grid[..., k + 1:] - self.grid[..., 1:-k]
            
            left = (x_expanded - self.grid[..., :-(k + 1)]) / left_denominator
            right = (self.grid[..., k + 1:] - x_expanded) / right_denominator
            bases = left * bases[..., :-1] + right * bases[..., 1:]
        return ops.reshape(bases, [ops.shape(x)[0], -1])

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.units,)

    def get_config(self):
        config = super(KANLinear, self).get_config()
        config.update({
            'units': self.units,
            'grid_size': self.grid_size,
            'spline_order': self.spline_order,
            'base_activation': self.base_activation_name,
            'grid_range': self.grid_range,
            'dropout': self.dropout_rate,
            'use_bias': self.use_bias,
            'use_layernorm': self.use_layernorm,
        })
        return config

    def get_build_config(self):
        return {"in_features": self.in_features}

    def build_from_config(self, config):
        self.build((None, config["in_features"]))

    @classmethod
    def from_config(cls, config):
        return cls(**config)