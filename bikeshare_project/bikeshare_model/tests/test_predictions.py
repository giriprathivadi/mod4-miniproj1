"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics import accuracy_score

from bikeshare_model.predict import make_prediction

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder


##################################################
# Test Weekday Imputer Trsnsformer
##################################################
def test_WeekdayImputer_transformer(sample_input_data):
    # Given
    transformer = WeekdayImputer(
        variable=config.model_config.weekday_var, date_var=config.model_config.date_var
    )

    assert np.isnan(sample_input_data.loc[7046, "weekday"])

    # sample_input_data = get_year_and_month(sample_input_data)

    print("----------weekday Before-------------")
    print(sample_input_data.loc[7046])
    print("-----------------------")
    # When
    subject = transformer.fit(sample_input_data).transform(sample_input_data)

    print("----------weekday After-------------")
    print(subject.loc[7046])
    print("-----------------------")

    # print('subject.loc[7046,weekday]:',subject.loc[7046,'weekday'])
    # Then
    assert subject.loc[7046, "weekday"] == "Wed"


##################################################
# Test Weathersit Imputer Trsnsformer
##################################################
def test_WeathersitImputer_transformer(sample_input_data):
    transformer = WeathersitImputer("weathersit")

    assert np.isnan(sample_input_data.loc[12230, "weathersit"])

    # sample_input_data = get_year_and_month(sample_input_data)

    print("----------weathersit Before-------------")
    print(sample_input_data.loc[12230])
    print("-----------------------")
    # When
    subject = transformer.fit(sample_input_data).transform(sample_input_data)

    print("----------weathersit After-------------")
    print(subject.loc[12230])
    print("-----------------------")

    # print('subject.loc[7046,weekday]:',subject.loc[7046,'weekday'])
    # Then
    assert subject.loc[12230, "weathersit"] == "Clear"


##################################################
# Test Month Mapper Trsnsformer
##################################################
def test_mnth_mapper_transformer(sample_input_data):
    transformer = Mapper("mnth", config.model_config.mnth_mappings)

    print("----------mnth Before-------------")
    print(sample_input_data.loc[12230, "mnth"])
    print("-----------------------")
    subject = transformer.fit(sample_input_data).transform(sample_input_data)
    print("----------mnth After-------------")
    print(subject.loc[12230, "mnth"])
    print("-----------------------")

    # print('subject.loc[7046,weekday]:',subject.loc[7046,'weekday'])
    # Then
    assert subject.loc[12230, "mnth"] == 9


##################################################
# Test Holiday Mapper Trsnsformer
##################################################
def test_holiday_mapper_transformer(sample_input_data):
    transformer = Mapper("holiday", config.model_config.holiday_mappings)

    print("----------holiday Before-------------")
    print(sample_input_data.loc[12230, "holiday"])
    print("-----------------------")
    subject = transformer.fit(sample_input_data).transform(sample_input_data)
    print("----------holiday After-------------")
    print(subject.loc[12230, "holiday"])
    print("-----------------------")

    assert subject.loc[12230, "holiday"] == 1


##################################################
# Test season Mapper Trsnsformer
##################################################
def test_season_mapper_transformer(sample_input_data):
    transformer = Mapper("season", config.model_config.season_mappings)

    print("----------season Before-------------")
    print(sample_input_data.loc[12230, "season"])
    print("-----------------------")
    subject = transformer.fit(sample_input_data).transform(sample_input_data)
    print("----------season After-------------")
    print(subject.loc[12230, "season"])
    print("-----------------------")

    assert subject.loc[12230, "season"] == 2


##################################################
# Test weathersit Mapper Trsnsformer
##################################################
def test_weathersit_mapper_transformer(sample_input_data):
    transformer_1 = WeathersitImputer("weathersit")
    sample_input_data = transformer_1.fit(sample_input_data).transform(
        sample_input_data
    )

    transformer = Mapper("weathersit", config.model_config.weathersit_mappings)

    print("----------weathersit Before-------------")
    print(sample_input_data.loc[7046, "weathersit"])
    print("-----------------------")
    subject = transformer.fit(sample_input_data).transform(sample_input_data)
    print("----------weathersit After-------------")
    print(subject.loc[7046, "weathersit"])
    print("-----------------------")

    assert subject.loc[7046, "weathersit"] == 3
