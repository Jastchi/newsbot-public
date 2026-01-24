#!/bin/bash
for file in "$@"; do
  if grep -EH "(# type: ignore|# noqa|# ruff: ignore|# ignore)" "$file"; then
    echo "Found type, linting, or general ignore comments in $file"
    echo "Please fix the underlying issue instead of suppressing warnings"
    exit 1
  fi
done
echo "No type, linting, or general ignore comments found"
