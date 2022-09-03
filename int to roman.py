class Solution:
    # @param A : integer
    # @return a strings
    def intToRoman(self, A):
        k=A
        roman = { 0:'', 1:'I', 2:'II', 3:'III', 4:'IV', 5:'V', 6:'VI', 7:'VII', 8:'VIII', 9:'IX',10:'X',
        20:'XX', 30: 'XXX', 40:'XL', 50:'L', 60:'LX', 70:'LXX', 80: 'LXXX', 90:'XC',100:'C', 3000:'MMM',
        200:'CC',300:'CCC', 400:'CD', 500:'D', 600:'DC', 700: 'DCC', 800:'DCCC',900:'CM', 1000:'M', 2000:'MM' }
        num = []
        while(k):
            num.append(k%10)
            k = int(k/10)
        ans = ''
        n = len(num)
        for i in range(0,len(num)):
            ans = ans + roman[num[n-i-1]*pow(10,n-i-1)] 
            
        return ans