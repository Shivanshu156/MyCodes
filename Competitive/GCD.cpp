int Solution::gcd(int A, int B) {
    int m = min(A,B), n = max(A,B);
    if(m==0)
    return n;
    if(n%m==0)
    return m;
    
    return gcd(n-m,m);
    
}
