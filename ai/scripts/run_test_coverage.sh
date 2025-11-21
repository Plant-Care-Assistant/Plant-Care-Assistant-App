#!/bin/bash -e

mkdir -p assets

coverage run --source="src/plant_care_ai" -m pytest "./tests"
coverage report -m
coverage-badge -f -o "${PWD}/assets/unit-coverage.svg"
