"""
Classes for system data and meta-data.
"""
import pandas as pd
import logging
import os
import re


logger = logging.getLogger(__name__)


class BaseBunch:
    """
    Container for CSV data and meta-data.

    Attributes:
        input_path (str): normalized path to the CSV file
        data_frame (str): pandas data frame of the data

    Properties:
        num_cols (int): number of columns in data
        num_rows (int): number of rows in data
    """
    def __init__(self, path):
        """
        Load CSV data into a pandas DataFrame.

        Args:
            path (str): system path to the CSV file
        """
        self.input_path = os.path.realpath(path)
        self.data_frame = self._load()
        self._process()

    @property
    def num_cols(self):
        return len(self.data_frame.columns)

    @property
    def num_rows(self):
        return len(self.data_frame.index)

    def _load(self):
        """
        Load CSV data into a Panda's DataFrame.
        """
        return pd.read_csv(self.input_path)

    def _process(self):
        """   Process CSV data.
        """
        raise NotImplementedError


class RADARBunch(BaseBunch):
    """
    Container for CSV data based on the RADAR homework.

    Pre-conditions:
        * CSV is assumed to have a column labeled TimeStamp, which is an int.
        * CSV is assumed to have a column labeled CycleCount, which is an int.
        * CSV is assumed to have source columns of the form: CAN trait
        * CSV is assumed to have object columns of the form: aObject[L].trait,
            where 'L' is a label refering to an object.
        * If a trait exists for an object, then it must exists for
            all objects.

    Attributes:
        source_columns (list): columns of the form CAN*
        source_traits  (list): traits for objects

        object_columns (list): columns of the form aObject[i].title
        object_traits  (list): traits for objects
        object_labels  (list): labels for objects

        column_time_stamp  (str): column name for TimeStamp
        column_cyclecount  (str): column name for CycleCount
    """
    def __init__(self, path):
        # source column(s) meta-data
        self.source_columns    = []            # list of source column names
        self.source_traits     = set()         # the 'T' in CAN T

        # object column(s) meta-data
        self.object_columns    = []            # list of object column names
        self.object_traits     = set()         # the 'T' in aObject[L].T
        self.object_labels     = set()         # the 'L' in aObject[L].T

        # extant column(s) meta-data
        self.column_time_stamp = 'TimeStamp'   # column name for TimeStamp
        self.column_cyclecount = 'CycleCount'  # column name for CycleCount
        super(RADARBunch, self).__init__(path)

    # source columns are assumed to match this regex
    _regex_source = re.compile(r'CAN\s*(?P<trait>.*)',
                               re.IGNORECASE | re.VERBOSE)
    _refmt_source = r'CAN {trait}'

    # object columns are assumed to match this regex
    _regex_object = re.compile(r'aObject\[(?P<label>\d+)\]\.(?P<trait>.+\..+)',
                               re.IGNORECASE | re.VERBOSE)
    _refmt_object = r'aObject[{label}].{trait}'

    def get_column_names_for_source(self, traits=None, extra=None):
        """
        Get the list of column names for the source.

        Parameters:
            traits (list[str]|str|None): object traits
            extra  (list[str]|None): extra columns

        Returns:
            list[str]: column names
        """
        if traits is None:
            traits = self.source_traits

        elif isinstance(traits, str):
            traits = [traits]

        indexer = []
        for trait in list(traits):
            indexer.append(self._refmt_source.format(trait=trait))

        if extra is not None:
            for col in extra:
                indexer.append(col)

        return indexer

    def get_column_names_for_object(self, label, traits=None, extra=None):
        """
        Get the list of column names for an object.

        Parameters:
            label  (int|str): object label
            traits (list[str]|str|None): object traits
            extra  (list[str]|None): extra columns

        Returns:
            list[str]: column names
        """
        if traits is None:
            traits = self.object_traits

        elif isinstance(traits, str):
            traits = [traits]

        indexer = []
        for trait in list(traits):
            indexer.append(self._refmt_object.format(label=label, trait=trait))

        if extra is not None:
            for col in extra:
                indexer.append(col)

        return indexer

    def get_column_names_for_many_objects(
            self, *labels, traits=None, extra=None):
        """
        Get the list of column names for many objects.

        Parameters:
            labels (list[int]|list[str]): object labels
            traits (list[str] or str or None): object traits
            extra  (list[str]): extra columns

        Returns:
            list[str]: column names
        """
        indexer = []
        for label in labels:
            indexer.extend(self.get_column_names_for_object(label, traits))

        if extra is not None:
            for col in extra:
                indexer.append(col)

        return indexer

    def get_subset_for_source(self, traits=None, extra=None):
        """
        Get the subset of data for the source.

        Parameters:
            traits (list[str]): object traits
            extra  (list[str]): extra columns

        Returns:
            DataFrame|Series: data subset
        """
        indexer = self.get_column_names_for_source(traits=traits, extra=extra)

        if len(indexer) == 1:
            return self.data_frame[indexer[0]]  # returns pd.Series instead
        else:
            return self.data_frame[indexer]

    def get_subset_for_object(self, label, traits=None, extra=None):
        """
        Get the subset of data for an object.

        Parameters:
            label  (int): object label
            traits (list[str]): object traits
            extra  (list[str]): extra columns

        Returns:
            DataFrame|Series: data subset
        """
        indexer = self.get_column_names_for_object(
            label, traits=traits, extra=extra)

        if len(indexer) == 1:
            return self.data_frame[indexer[0]]  # returns pd.Series instead
        else:
            return self.data_frame[indexer]

    def get_subset_for_many_objects(self, *labels, traits=None, extra=None):
        """
        Get the subset of data for many objects.

        Parameters:
            labels (list): object labels
            traits (list[str]): object traits
            extra  (list[str]): extra columns

        Returns:
            DataFrame|Series: data subset
        """
        indexer = self.get_column_names_for_many_objects(
            *labels, traits=traits, extra=extra)

        if len(indexer) == 1:
            return self.data_frame[indexer[0]]  # returns pd.Series instead
        else:
            return self.data_frame[indexer]

    def _process(self):
        # examine all dataframe columns
        for col in self.data_frame.columns:

            # process object columns
            match = self._regex_object.match(col)
            if match:
                self.object_columns.append(col)
                label = int(match.group('label'))
                trait = str(match.group('trait'))
                self.object_labels.add(label)
                self.object_traits.add(trait)

            # process source columns
            else:
                match = self._regex_source.match(col)
                if match:
                    self.source_columns.append(col)
                    trait = str(match.group('trait'))
                    self.source_traits.add(trait)

                # process extant columns
                else:
                    if col not in [
                        self.column_time_stamp,
                        self.column_cyclecount
                    ]:
                        raise RuntimeError('unknown column: %s' % col)

        # make a more meaningful time column
        # the TimeStamp column looks like microseconds...
        t0 = self.data_frame.ix[0, self.column_time_stamp]
        self.data_frame['TimeStamp[s]']  = \
            (self.data_frame[self.column_time_stamp] - t0) * 1e-6

        # remove duplicates and sort
        self.object_labels = sorted(list(self.object_labels))
        self.object_traits = sorted(list(self.object_traits))
        self.source_traits = sorted(list(self.source_traits))

        # log some info
        self._debug()

    def _debug(self):
        """
        Output CSV meta-data to logger.
        """
        logger.debug('found columns: %d', self.num_cols)
        logger.debug('found sources: %d', len(self.source_columns))
        logger.debug('found objects: %d', len(self.object_columns))
        logger.debug('found olabels: %d', len(self.object_labels))
        logger.debug('found otraits: %d', len(self.object_traits))

        # log columns
        for key in self.source_traits:
            col = self._refmt_source.format(trait=key)
            logger.debug('column (source): %s', col)

        for key in self.object_traits:
            col = self._refmt_object.format(label='i', trait=key)
            logger.debug('column (object): %s', col)
