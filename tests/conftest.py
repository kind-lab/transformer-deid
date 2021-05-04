# Fixtures used by various tests
# Fixtures are a good way to
#  - load and share data across tests
#  - inject dependencies into tests
# We use environment variables in fixtures.
# If not present, we use default values.
import json
import os

import pytest

from transformers import DistilBertTokenizerFast


@pytest.fixture(scope="session")
def radiology_reports():
    """
    Load the test radiology reports.
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    text_path = os.path.join(dir_path, 'fake-data', 'radiology-report')

    reports_list = os.listdir(text_path)
    reports_list.sort()

    assert len(reports_list) == 7

    # load reports
    reports = {}
    for f in reports_list:
        with open(os.path.join(text_path, f), 'r') as fp:
            reports[f] = ''.join(fp.readlines())

    assert len(reports) == 7

    return reports

@pytest.fixture(scope="session")
def tokenizer():
    return DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')