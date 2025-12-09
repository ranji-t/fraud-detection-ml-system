# Third party Imports
from pydantic import BaseModel, PositiveInt, PositiveFloat, NonNegativeInt


class ModelConfig(BaseModel):
    input_size: PositiveInt
    hidden1_size: PositiveInt
    bottleneck_size: PositiveInt
    hidden2_size: PositiveInt
    output_size: PositiveInt


class SchedulerConfig(BaseModel):
    init_value: PositiveFloat
    peak_value: PositiveFloat
    warmup_steps: NonNegativeInt
    decay_steps: NonNegativeInt
    end_value: PositiveFloat
    exponent: PositiveFloat = 1.0


class TrainingConfig(BaseModel):
    shuffle_seed: PositiveInt = 101
    epochs: PositiveInt = 100
    mini_batches: PositiveInt = 5
    batch_size: PositiveInt = 32
    schedulers: SchedulerConfig


class AutoEncoderConfig(BaseModel):
    seed: PositiveInt
    model: ModelConfig
    training: TrainingConfig
