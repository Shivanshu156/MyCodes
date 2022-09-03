int Solution::divide(int A, int B) {
    long long n=abs(long(A)), m=abs(B), t=0, sign;
    long long q=0;
    sign = ((A>0 && B>0)||(A<0 && B<0))? 1 : -1;
    if(A==INT_MIN && B==-1) return INT_MAX;
    if(A==0) return 0;
    if(B==0) return INT_MAX;
    for(int i=31;i>=0;i--)
    { if( (t+ (m<<i)) <= n )
        {t += (m<<i);
        q += 1LL<<i;
            // cout<<"i here is "<<i<<endl;
        }
    }
    // cout<<"sign is "<<sign<<endl;
    // cout<<"q is "<<q<<endl;
    
    q = sign*q;
    if(q>INT_MAX) return INT_MAX;
    if(q<INT_MIN) return INT_MAX;
    return q;
}