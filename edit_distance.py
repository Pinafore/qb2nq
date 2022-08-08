def printChanges(s1, s2, dp):
     
    i = len(s1)
    j = len(s2)
     
   # Check till the end
    while(i > 0 and j > 0):
         
        # If characters are same
        if s1[i - 1] == s2[j - 1]:
            i -= 1
            j -= 1
             
        # Replace
        elif dp[i][j] == dp[i - 1][j - 1] + 1:
            print("change", s1[i - 1],
                      "to", s2[j - 1])
            j -= 1
            i -= 1
             
        # Delete
        elif dp[i][j] == dp[i - 1][j] + 1:
            print("Delete", s1[i - 1])
            i -= 1
             
        # Add
        elif dp[i][j] == dp[i][j - 1] + 1:
            print("Add", s2[j - 1])
            j -= 1


def editDP(s1, s2):     
    len1 = len(s1)
    len2 = len(s2)
    dp = [[0 for i in range(len2 + 1)]
             for j in range(len1 + 1)]
     
    # Initialize by the maximum edits possible
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
     
    # Compute the DP Matrix
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
             
            # If the characters are same
            # no changes required
            if s2[j - 1] == s1[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                 
            # Minimum of three operations possible
            else:
                dp[i][j] = 1 + min(dp[i][j - 1],
                                   dp[i - 1][j - 1],
                                   dp[i - 1][j])
                                    
    # Print the steps
    #printChanges(s1, s2, dp)
    return dp[-1][-1]
def wfi_levenshtein(string_1, string_2):
    """
    Calculates the Levenshtein distance between two strings.
    This version uses an iterative version of the Wagner-Fischer algorithm.
    Usage::
        >>> wfi_levenshtein('kitten', 'sitting')
        3
        >>> wfi_levenshtein('kitten', 'kitten')
        0
        >>> wfi_levenshtein('', '')
        0
    """
    if string_1 == string_2:
        return 0

    len_1 = len(string_1)
    len_2 = len(string_2)

    if len_1 == 0:
        return len_2
    if len_2 == 0:
        return len_1

    if len_1 > len_2:
        string_2, string_1 = string_1, string_2
        len_2, len_1 = len_1, len_2

    d0 = [i for i in range(len_2 + 1)]
    d1 = [j for j in range(len_2 + 1)]

    for i in range(len_1):
        d1[0] = i + 1
        for j in range(len_2):
            cost = d0[j]

            if string_1[i] != string_2[j]:
                # substitution
                cost += 1

                # insertion
                x_cost = d1[j] + 1
                if x_cost < cost:
                    cost = x_cost

                # deletion
                y_cost = d0[j + 1] + 1
                if y_cost < cost:
                    cost = y_cost

            d1[j + 1] = cost

        d0, d1 = d1, d0
    #printChanges(string_1, string_2, d0);
    return d0[-1]

#import pylev
#import editdistance
def edit_distance(a,b):
  #a = "Of the three officers arrested for dumping the chemical waste, a freight industry is still not clear what is the waste from a plastics factory."
  #b = "three officers of a freight industry were arrested for dumping the chemical waste, but it is still not clear what the waste is that came from a plastics factory."
  a = a.split(" ")
  b = b.split(" ")
  delete=0
  insert=0
  shift=0
  for i in a:
      if(i not in b) :
          #print(i)
          delete=delete+1
  for i in a:
      if(i in b) :
        if (a.index(i)!=b.index(i)) :
          #print(i)
          shift=shift+1
  for i in b:
      if(i not in a) :
          insert=insert+1
  HTER=insert+delete+shift
  #print(wfi_levenshtein(a,b))
  #print(editDP(a,b))
  return editDP(a,b)
  #print(pylev.damerau_levenshtein(a,b))
  #print(editdistance.eval(a, b))
  #print(HTER)
