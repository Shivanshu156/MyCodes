//  strstr - locate a substring ( needle ) in a string ( haystack ). 

int Solution::strStr(const string A, const string B) {
    if(B.size()==0 || A.size()==0) return -1;
    bool check = false;
    for(int i=0; i<A.size(); i++)
    {if(A[i]==B[0])
        {   check = true;
            for(int j=0; j<B.size();j++)
            if(A[i+j]!=B[j]) {check = false; break;}
            if(check) return i;
        }
        
    }
    return -1;
}