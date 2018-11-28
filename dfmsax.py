import numpy as np
import pandas as pd
import saxpy.msax as msax

class DataFrameInfo:
    """
    This class is used for encapsulating information for a time-series 
    contained in a pandas.DataFrame object.
    """
    def __init__(self, dataframe, time_col, value_col) :
        assert isinstance(dataframe, pd.DataFrame)

        self.df = dataframe
        self.t_col = time_col
        self.v_col = value_col

    def __len__(self) :
        return len(self.df)

    def setDuration(self, duration) :
        t_end = self.df[self.t_col].min() + duration
        if (self.df[self.t_col].max() >= t_end) :
            self.df = self.df[self.df[self.t_col] <= t_end]
        else :
            loc = self.df.index.max()+1
            self.df.loc[loc, self.t_col] = t_end

class DataFrameInfoMSAX(msax.MSAX) :
    """
    This class is for using the Modified Symbolic Aggregate approXimation (MSAX) method 
    with a DataFrameInfo object.  In short, this translates a series of data to a string, 
    which can then be compared with other such strings using a lookup table.
    """

    def __init__(self, wordSize = 8, alphabetSize = 7, epsilon = 1e-6) :
        super().__init__(wordSize, alphabetSize, epsilon)

    def normalize(self, x):
        """
        Function will normalize an array (give it a mean of 0, and a
        standard deviation of 1) unless it's standard deviation is below
        epsilon, in which case it returns an array of zeros the length
        of the original array.
        """
        if isinstance(x, DataFrameInfo) :
            d = x.df.copy()
            d[x.v_col] = (d[x.v_col] - d[x.v_col].mean()) / d[x.v_col].std()
            x.df = d
            return x
        else:
            return super().normalize(x)

    def to_PAA(self, x):
        """
        Function performs Piecewise Aggregate Approximation on data set, reducing
        the dimension of the dataset x to w discrete levels. returns the reduced
        dimension data set, as well as the indices corresponding to the original
        data for each reduced dimension
        """
        if isinstance(x, DataFrameInfo) :
            t_start = x.df[x.t_col].min()
            t_max = x.df[x.t_col].max()
            t_duration =  t_max - t_start
            t_step = t_duration / self.wordSize
            paa = [None] * self.wordSize
            indices = [None] * self.wordSize

            for i in range(self.wordSize) :
                t_stop = t_start + t_step
                piece = x.df[(x.df[x.t_col] >= t_start) & (x.df[x.t_col] < t_stop)]
                paa[i] = piece[x.v_col].mean()
                indices[i] = (t_start, t_stop)
                #paa.append(piece[x.v_col].mean())
                #indices.append((t_start, t_stop))
                t_start += t_step

            return (np.array(paa), indices)
        else :
            return super().to_PAA(x)
