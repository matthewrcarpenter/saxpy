import numpy as np
import math
from saxpy.saxpy import SAX

class MSAX(SAX) :
    """
    This implements a Modified Symbolic Aggregate approXimation (MSAX) method.

    In general, SAX translates a series of data to a string, which can then be 
    compared with other such strings using a lookup table. 'Normal' SAX defines 
    adjacent letters (e.g. 'A' and 'B') to have zero distance. Modified SAX (MSAX)
    defines a non-zero distance between adjacent letters.

    For example, given the two strings 'aaa' and 'bba', 'normal' SAX gives a zero
    distance between the two. MSAX will gives a non-zero distance, and that distance
    will also be greater than the distance between the two strings 'aaa' and 'baa'.    
    """

    def __init__(self, wordSize = 8, alphabetSize = 7, epsilon = 1e-6) :
        super().__init__(wordSize, alphabetSize, epsilon)
        self.breakpoints['21'] = [-1.67, -1.31, -1.07, -0.88, -0.71, -0.57, -0.43, -0.3 , -0.18, -0.06,  0.06,  0.18,  0.3 ,  0.43,  0.57,  0.71,  0.88,  1.07, 1.31,  1.67]
        self.letterVals = self.breakpoints[str(alphabetSize+1)]
        self.rebuild_letter_compare_dict()

    def rebuild_letter_compare_dict(self) :
        """
        Builds up the lookup table to determine numeric distance between two letters
        given an alphabet size.  Entries for both 'ab' and 'ba' will be created
        and will have have identical values. 
        
        Unlike 'normal' SAX, adjacent letters (e.g. 'ab' or 'cb') will have a non-zero distance.
        """
        number_rep = range(0,self.alphabetSize)
        letters = [chr(x + self.aOffset) for x in number_rep]
        self.compareDict = {}
        for i in range(0, len(letters)):
            for j in range(0, len(letters)):
                #print(f'{i},{j}')
                if (number_rep[i] == number_rep[j]) :
                    self.compareDict[letters[i]+letters[j]] = 0
                else:
                    high_num = np.max([number_rep[i], number_rep[j]])
                    low_num = np.min([number_rep[i], number_rep[j]])
                    self.compareDict[letters[i]+letters[j]] = self.letterVals[high_num] - self.letterVals[low_num]
                #print(f'{self.compareDict}')
