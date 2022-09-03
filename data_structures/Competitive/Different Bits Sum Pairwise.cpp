int Solution::cntBits(vector<int> &A) {
    long long n=A.size(), ans = 0;
    
    for(int i=0;i<32;i++)
    {long long one_count=0;
     for(long long j=0;j<n;j++)
     if( (A[j]>>i)&1 ) one_count++;
     
     ans+= (((one_count%1000000007)*((n-one_count)%1000000007))%1000000007)*2;
     ans%= 1000000007;
    }

    return int(ans);
}