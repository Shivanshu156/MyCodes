class Solution:
    # @param A : string
    # @return an integer
    def romanToInt(self, A):
        num = { 'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        n = len(A)
        ans = num[A[n-1]]
        for i in range(2,len(A)+1):
            if num[A[n-i]] < num[A[n-i+1]] :
                ans = ans - num[A[n-i]]
            else:
                ans = ans + num[A[n-i]]
            
        return ans