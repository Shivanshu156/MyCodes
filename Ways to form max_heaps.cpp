map<pair<int,int>, long long int> S;
#define mod 1000000007
long long int C(int n, int r)
{   vector<long long int> A(r+1,0);
    A[0] = 1;
    for(int i=1;i<=n;i++)
    for(int j = min(i,r);j>0;j--)
    A[j] = (A[j]+A[j-1])%mod;
    return A[r];
}


int Solution::solve(int A) {
    
    if(A==1) return 1;
    if(A==0) return 1;
    // if(S.find(A)!=S.end()) return S[A];
    long long int ans, L, R, depth, last_n;
    depth = int(log2(A))+1;
    last_n = A - pow(2,depth-1) + 1;
    if(last_n>=pow(2,depth-2))  L = pow(2,depth-1) - 1;
    else L = pow(2, depth-2) + last_n -1;
    R = A-1-L;
    ans = ((C(A-1, L)%1000000007* (solve(L)%1000000007))%1000000007 * (solve(R)%1000000007))%1000000007;
    // S.insert(make_pair(A, ans));
    return ans%1000000007;
    
    // return C(19,12);
    
}