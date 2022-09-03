int Solution::isPalindrome(string A) {
    int n = A.size();
    char p,q;
    int i=0,j=n-1, count = 0;
    // if(n==1) return 1;
    
    for(int i=0;i<A.size(); i++)
    if(isalnum(A[i])) count++;
    if(count==0 || count == 1) return 1;
    while(i<=j)
    {   
        // cout<<"here"<<endl;
        while(i<n-1 && !isalnum(A[i]))
        i++;
        // cout<<" i is "<<i;
        while(j>0 && !isalnum(A[j]))
        j--;
        // cout<<" j is "<<j<<endl;
        
        // cout<<"p is "<<p<<" q is "<<q<<endl;
        p=A[i]; q=A[j];
        if(!(isalnum(p) && isalnum(q))) return 0;
        if(tolower(p)!=tolower(q)) return 0;
        i++; j--; 
        
    }
    return 1;
}
