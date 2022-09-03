// Input 1:
//     A = "the sky is blue"
// Output 1:
//     "blue is sky the"

// Input 2:
//     A = "this is ib"
// Output 2:
//     "ib is this"


string Solution::solve(string A) {
    // cout<<"rejhfbjerbvknkv\n";
    vector<string> B;
    string word;
    // cout<<"A.size() is "<<A.size();
    if(A.size()==0) return A;
    for(int i=0;i<A.size();i++)
    {   
        if(A[i]!=' ')
        word.push_back(A[i]);
        if(A[i]==' ') {B.push_back(word); word.clear();}
    }
    if(word!="")
    B.push_back(word);
    // cout<<B[0];
    // reverse(B.begin(), B.end());
    string ans;
    for(int i=0; i<B.size()-1;i++)
    ans += B[B.size()-1-i] + ' ';
    ans = ans + B[0];  
    return ans;
}
