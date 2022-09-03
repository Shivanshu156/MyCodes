long long int fact(int a)
{   if(a==0) return 1;
    if(a==1) return 1;
    return ((a%1000003)*(fact(a-1)%1000003))%1000003;
    
}

int permutation(string A)
{
    char pivot = A[0];
    int n = A.size(), rank=0, count=0;
    if(n==1) return 0;
    string B = A.substr(1,n-1);
    for(int i=0; i<B.size();i++)
    if (B[i]<pivot) count++;
    rank = rank%1000003 + (count*fact(B.size()))%1000003 + permutation(B)%1000003;
    
    return rank;
    
    
}

int Solution::findRank(string A) {
    int ans;
    ans = permutation(A)+1;
    ans %= 1000003;
    return ans;
    
}
