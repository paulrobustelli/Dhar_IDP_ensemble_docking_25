import warnings
import functools
import os
import numpy as np 
import re

def num_str(s, return_str=True, return_num=True):
    s = ''.join(filter(str.isdigit, s))
    if return_str and return_num:
        return s, int(s)
    if return_str:
        return s
    if return_num:
        return int(s)

def sort_strs(strs: list, max=False, indexed: bool=False):
    
    """ strs ::: a list or numpy array of strings.
        max ::: bool, to sort in terms of greatest index to smallest.
        indexed ::: bool, whether or not to filter out strings that don't contain digits.
                    if set to False and string list (strs) contains strings without a digit, function 
                    will return unsorted string list (strs) as an alternative to throwing an error."""
    
    # we have to ensure that each str in strs contains a number otherwise we get an error
    check = np.vectorize(lambda s : any(map(str.isdigit, s)))
    
    if isinstance(strs, list):
        strs = np.array(strs)
    # the indexed option allows us to filter out strings that don't contain digits.
    ## This prevents an error
    if indexed:
        strs = strs[check(strs)]
    #if indexed != True, then we don't filter the list of input strings and simply return it
    ##because an attempt to sort on indices (digits) that aren't present results in an error
    else:
        if not all(check(strs)):
            
            warnings.warn("Not all strings contain a number, returning unsorted input list to avoid throwing an error. "
                        "If you want to only consider strings that contain a digit, set indexed to True ")
            return strs
    get_index = np.vectorize(functools.partial(num_str, return_str=False, return_num=True))
    indices = get_index(strs).argsort()
    if max:
        return strs[np.flip(indices)]
    else:
        return strs[indices]

def lsdir(dir, keyword: "list or str" = None,
          exclude: "list or str" = None,
          indexed:bool=False):
    """ full path version of os.listdir with files/directories in order
        dir ::: path to a directory (str), required
        keyword ::: filter out strings that DO NOT contain this/these substrings (list or str)=None
        exclude ::: filter out strings that DO contain this/these substrings (list or str)=None
        indexed ::: filter out strings that do not contain digits.
                    Is passed to sort_strs function (bool)=False"""
    if dir[-1] == "/":
        dir = dir[:-1] # slicing out the final '/'
    listed_dir = os.listdir(dir) # list out the directory 
    if keyword is not None:
        listed_dir = keyword_strs(listed_dir, keyword) # return all items with keyword
    if exclude is not None:
        listed_dir = keyword_strs(listed_dir, keyword=exclude, exclude=True) # return all items without excluded str/list
    # Sorting (if possible) and ignoring hidden files that begin with "." or "~$"
    return [f"{dir}/{i}" for i in sort_strs(listed_dir, indexed=indexed) if (not i.startswith(".")) and (not i.startswith("~$"))] 

def keyword_strs(strs: list, keyword: "list or str", exclude: bool = False):
    
    if isinstance(keyword, str): # if the keyword is just a string 
        if exclude:
            filt = lambda string: keyword not in string
        else:
            filt = lambda string: keyword in string
    else:
        if exclude:
            filt = lambda string: all(kw not in string for kw in keyword)
        else:
            filt = lambda string: all(kw in string for kw in keyword)
    return list(filter(filt, strs))

def get_filename(filepath):
    """ returns a string of the file name of a filepath """
    return filepath.split('/')[-1].split('.')[0]

def chk_mkdir(newdir, nreturn=False):
    """ Checks and makes the directory if it doesn't exist. If True, nreturn will return the new file path. """
    isExist = os.path.exists(newdir)
    if not isExist:
        os.mkdir(newdir)
    if nreturn:
        return newdir
    

def rm_path(x): 
    return x.split("/")[-1]

def anymatch(string, strings, rm_path_=True):
    
    if rm_path_:
        string = rm_path(string)
    return any(map(string.count,iter(strings)))

def itermatch(string_list, strings, rm_path_=False):
    func = functools.partial(anymatch, strings=strings, rm_path_=rm_path_)
    return list(filter(func, string_list))

def str_csv(string:str, replacement=" , ", split=False):
    return re.sub("\s+", replacement, string.strip())
