#define mod 1000000007
long long int C(int n, int r)
{   vector<long long int> A(r+1,0);
    A[0] = 1;
    for(int i=1;i<=n;i++)
    for(int j = min(i,r);j>0;j--)
    A[j] = (A[j]+A[j-1])%mod;
    return A[r];
}