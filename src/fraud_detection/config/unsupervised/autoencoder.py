# Third party Imports
from pydantic import BaseModel, PositiveInt, PositiveFloat


class ModelConfig(BaseModel):
    input_size: PositiveInt
    hidden1_size: PositiveInt
    bottleneck_size: PositiveInt
    hidden2_size: PositiveInt
    output_size: PositiveInt


class LinearScheduleConfig(BaseModel):
    init_value: PositiveFloat
    end_value: PositiveFloat
    transition_steps: PositiveInt


class CosineScheduleConfig(BaseModel):
    init_value: PositiveFloat
    decay_steps: PositiveInt
    alpha: PositiveFloat


class SchedulerConfig(BaseModel):
    boundaries: list[PositiveInt]
    linear_schedule: LinearScheduleConfig
    cosine_decay_schedule: CosineScheduleConfig


class TrainingConfig(BaseModel):
    epochs: PositiveInt = 100
    batch_size: PositiveInt = 32
    schedulers: SchedulerConfig


class AutoEncoderConfig(BaseModel):
    seed: PositiveInt
    model: ModelConfig
    training: TrainingConfig
