NAME = cinterview
NOSE = nosetests --verbosity 2
CCMD = find . -type f -name \*.pyc             -or \
              -type d -name __pycache__        -or \
              -type d -name .ipynb_checkpoints

.PHONY: all
all: help

.PHONY: help
help:
	-@echo "Targets"
	-@echo "======="
	-@echo "  * make conda : create conda environment"
	-@echo "  * make tests : run tests"
	-@echo "  * make clean : remove files (dry run)"
	-@echo "  * make force : remove files"

.PHONY : tests
tests:
	-export NOSE_REDNOSE=1; ${NOSE} -w .

.PHONY: clean
clean:
	-@echo "Looking for files..."
	-@${CCMD}
	-@echo "Done"
	-@echo "\033[31mTo remove files, use: make force\033[0m"

.PHONY: force
force:
	-@echo "Looking for files..."
	-@${CCMD} | xargs -I xxx rm -rfv xxx
	-@echo "Done"

.PHONY: conda
conda:
	-conda remove --name $(NAME) --all --yes
	conda env create --file ./environment.yml
	conda env list
	conda env export --name $(NAME)
	@echo "\033[31mTo activate use: source activate $(NAME)\033[0m"
	@echo ""

