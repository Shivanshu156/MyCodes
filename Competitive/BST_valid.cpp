int value, prevalue;

bool check(TreeNode* root)
{   bool left, right;
    if(root==NULL) return true;
    left = check(root->left);
    prevalue = value;
    value = root->val;
    if(value<=prevalue) return false;
    right = check(root->right);
    return (left&&right);
}
int Solution::isValidBST(TreeNode* A) {
    
    TreeNode *root = A;
    while(root->left!=NULL) root=root->left;
    value = root->val-1;
    // cout<<"value is "<<value;
    return check(A);
    // return true;
}