int Solution::atoi(const string A) {
    string A1; bool flag = true;
    for(int i=0;i<A.size();i++)
    A1.push_back(A[i]);
    // Removed front spaces from string 
    while(A1[0]==' ')
        A1.erase(0, 1);
    while(A1[0]=='+')
        A1.erase(0, 1);
    int n = A1.size(), i=0; string word;
    if(A1[0]=='-') {word.push_back(A1[0]); i++;}
    while( A1[i]!=' ' && isdigit(A1[i]))
    {    word.push_back(A1[i]); i++;
        // cout<<"here in while";
        if(stoi(word)> INT_MAX/10 && isdigit(A1[i])) return INT_MAX;
        if(stoi(word)< INT_MIN/10 && isdigit(A1[i])) return INT_MIN;
        if(stoi(word)== INT_MAX/10 && isdigit(A1[i])>7) return INT_MAX;
        if(stoi(word)== INT_MIN/10 && isdigit(A1[i])>8) return INT_MIN;
    }
    // if(!isdigit(A1[i]) && A1[i]!=' ' && i<A1.size()){cout<<"check1"; 
    // return 0;}
    for(int i=1;i<word.size(); i++)
    if(!isdigit(word[i])) {flag = false; //cout<<"check2";
        
    }
    if(!isdigit(word[0]) && word[0]!='-'){//cout<<"check3"; 
    flag = false;}
    if(word.size()==1 && !isdigit(word[0])) {//cout<<"check4"; 
    return 0;}
    if(flag) return(stoi(word));
    else return 0;
}



class Solution {
    public:
        int atoi(const string &str) {
            int sign = 1, base = 0, i = 0;
            while (str[i] == ' ') { i++; }
            if (str[i] == '-' || str[i] == '+') {
                sign = (str[i++] == '-') ? -1 : 1; 
            }
            while (str[i] >= '0' && str[i] <= '9') {
                if (base >  INT_MAX / 10 || (base == INT_MAX / 10 && str[i] - '0' > 7)) {
                    if (sign == 1) return INT_MAX;
                    else return INT_MIN;
                }
                base  = 10 * base + (str[i++] - '0');
            }
            return base * sign;
        }
};