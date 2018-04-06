MAKE_PATH="C:\Program Files (x86)\GnuWin32\bin\make.exe"
PIP_INSTALL=pip install --no-cache-dir
ENV=tsi-environment
ENV_PATH=C:\Miniconda\envs\tsi-environment

all: intro deps
	conda activate $(ENV) && python spheres.py

intro:
	@echo -----------------------------------
	@echo    WELCOME TO TSI / stars / 2018
	@echo    new_sphere_era
	@echo -----------------------------------
	@echo    !!! REQUIRES:
	@echo    !!! conda (which should handle Python, Microsoft Visual C++ Compiler for Python 2.7, pip, and other system deps)
	@echo    !!! https://avbin.github.io/AVbin/Download.html
	@echo    !!!
	@echo    !!! AVBin
	@echo    !!! https://avbin.github.io/AVbin/Download.html
	@echo -----------------------------------

deps:
	conda env update -n $(ENV)

deps_clean:
	conda env remove -n $(ENV)
	del /Q $(ENV_PATH)

deps_update:
	conda update -n base conda

# WARNING: Failed (mostly pip) experiments; use conda instead
qutip_conda_env:
	cd ../ && conda create -n $(ENV) -p $(ENV_PATH) && conda activate $(ENV) && pip install crayons && conda env update -n $(ENV)

qutip_pip: qutip_deps qutip_install qutip_post
	@echo Installed qutip with pip

qutip_deps:
	$(PIP_INSTALL) numpy
	$(PIP_INSTALL) scipy
	$(PIP_INSTALL) matplotlib
	$(PIP_INSTALL) cython

qutip_install:
	$(PIP_INSTALL) qutip

qutip_post:
	$(PIP_INSTALL) sympy
	$(PIP_INSTALL) mpmath
	$(PIP_INSTALL) pyglet
	$(PIP_INSTALL) crayons
	$(PIP_INSTALL) vpython

clean:
	conda deactivate $(DEPS_ENV)
	del $(ENV_PATH)
