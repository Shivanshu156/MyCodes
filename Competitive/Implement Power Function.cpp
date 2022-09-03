nt mypow(long long int x,long long int n,long long int d)
{   long long int ans, x_2;
    if(x==0) return 0;
    if(n==0) return 1%d;
    if(n==1 && x > 0) return x%d;
    if(n==1 && x < 0) return (d + x%d)%d;    
    if(n%2==0)
    {   x_2 = ((x%d)*(x%d))%d;
        // cout<<"n is "<<n<<" x square is "<<(x%d)*(x%d)<<endl;
        ans =  mypow(x_2,n/2,d);
        
        // cout<<" and answer is "<<ans<<endl;
    }
    if(n%2==1 && n!=1)
    {   x_2 = ((x%d)*(x%d))%d;
        // cout<<"n is "<<n<<" x square is "<<x_2<<endl;
        ans = (mypow(x_2,(n-1)/2,d)*(x%d))%d;
        // cout<<"n is "<<n;
        // cout<<" and ans is "<<ans<<endl;
    }
    return ans;
}

int Solution::pow(int x, int n, int d) {
    
    int ans;
    ans = mypow(x,n,d);
    if(ans < 0 )
    ans = d - (ans*-1)%d;
    
    return ans;
}
