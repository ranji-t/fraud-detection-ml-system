# Third party imports
from pydantic import (
    BaseModel,
    FilePath,
    field_validator,
    PositiveFloat,
    PositiveInt,
    NonNegativeInt,
)


class ClassWeightsConfig(BaseModel):
    neg_weight: PositiveFloat = 1.0
    pos_weight: PositiveFloat = 1.0


class DataConfig(BaseModel):
    input_path: FilePath


class LrScheduleConfig(BaseModel):
    boundaries_and_scales: dict[PositiveInt, PositiveFloat]
    init_value: PositiveFloat


class ReadCsvConfig(BaseModel):
    ignore_errors: bool = False
    infer_schema_length: PositiveInt = 1000


class TestSplitConfig(BaseModel):
    random_state: NonNegativeInt = 42
    train_size: PositiveFloat = 0.2

    @field_validator("train_size", mode="after")
    def validate_test_size(cls, value):
        if value <= 0 or value >= 1:
            raise ValueError("train_size must be between 0 and 1")
        return value


class ValidSplitConfig(BaseModel):
    random_state: NonNegativeInt = 42
    train_size: PositiveFloat = 0.2

    @field_validator("train_size", mode="after")
    def validate_valid_size(cls, value):
        if value <= 0 or value >= 1:
            raise ValueError("train_size must be between 0 and 1")
        return value


class SplitConfig(BaseModel):
    test_split: TestSplitConfig
    valid_split: ValidSplitConfig


class TrainConfig(BaseModel):
    epochs: PositiveInt = 50


class Config(BaseModel):
    class_weights: ClassWeightsConfig
    data: DataConfig
    lr_schedule: LrScheduleConfig
    read_csv: ReadCsvConfig
    split: SplitConfig
    train: TrainConfig
