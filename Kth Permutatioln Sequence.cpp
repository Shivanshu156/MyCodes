// The set [1,2,3,…,n] contains a total of n! unique permutations.

// By listing and labeling all of the permutations in order,
// We get the following sequence (ie, for n = 3 ) :

// 1. "123"
// 2. "132"
// 3. "213"
// 4. "231"
// 5. "312"
// 6. "321"
// Given n and k, return the kth permutation sequence.

// For example, given n = 3, k = 4, ans = "231"

//  Good questions to ask the interviewer :
// What if n is greater than 10. How should multiple digit numbers be represented in string?
//  In this case, just concatenate the number to the answer.
// so if n = 11, k = 1, ans = "1234567891011" 
// Whats the maximum value of n and k?
//  In this case, k will be a positive integer thats less than INT_MAX.
// n is reasonable enough to make sure the answer does not bloat up a lot. 







long long int fact(int x){
    long long int f;
    if(x==0)    return 1;
    if(x == 1) return x;
    
    f = x * fact(x-1);
    if(f>INT_MAX)   return INT_MAX;
    return f;
}


void perm(vector<int> &A, int k, vector<int> &ans)
{   
   int n = A.size();
    if(n==2){  if(k%2==0) {ans.push_back(A[0]); ans.push_back(A[1]); return;} 
                else{ans.push_back(A[1]); ans.push_back(A[0]); return;}     }
    int index;
    long long int f = fact(n-1);

    index = (k)/f;
    ans.push_back(A[index]);
    A.erase(A.begin()+index);
    if(n-1 > 2)
    k= k%f;
    perm(A, k, ans);
    
}
string Solution::getPermutation(int A, int B) {
    // if(A>13) return to_string(INT_MAX);
    if(A==1) return to_string(1);
    if(B>fact(A)) return to_string(INT_MAX);
    vector<int> A1, ans;
    for(int i=0;i<A; i++)
    A1.push_back(i+1);
    // cout<<fact(13);
    perm(A1, B-1, ans); 
    string sol="";
    for(int i=0;i<ans.size();i++)
    sol+=to_string(ans[i]);
    
    return sol;
    
}
