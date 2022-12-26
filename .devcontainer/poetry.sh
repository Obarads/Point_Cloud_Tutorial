HOME_PATH="/home/user/"

# Install poetry
curl -sSL https://install.python-poetry.org | python3.10 -
POETRY_BIN_DIR_PATH=${HOME_PATH}.local/bin
POETRY_ACTIVATION='export PATH="${POETRY_BIN_DIR_PATH}:$PATH"'
echo $POETRY_ACTIVATION >> ${HOME_PATH}/.bashrc
${POETRY_BIN_DIR_PATH}/poetry config virtualenvs.in-project true

# Install python env with poetry
cd $(dirname ${0})
cd ../
${POETRY_BIN_DIR_PATH}/poetry install ${1}
PYTHON_ACTIVATION="source $(pwd)/.venv/bin/activate"
echo $PYTHON_ACTIVATION >> ~/.bashrc
