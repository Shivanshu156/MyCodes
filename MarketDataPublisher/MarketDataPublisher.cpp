#include <iostream>
using namespace std;

struct node
{
	long int C_id, S_price, bf;
	node *left;
	node *right;
};

void MergeSort(long int a[][2], long int l, long int r)
{
	if (l < r)
	{
		long int m = (r + l) / 2;
		MergeSort(a, l, m);
		MergeSort(a, m + 1, r);
		//////Merging Starts
		long int n1 = m - l + 1, n2 = r - m;
		long int L[n1][2], R[n2][2];

		for (long int i = 0; i < n1; i++)
		{
			L[i][0] = a[l + i][0];
			L[i][1] = a[l + i][1];
		}
		for (long int j = 0; j < n2; j++)
		{
			R[j][0] = a[m + 1 + j][0];
			R[j][1] = a[m + 1 + j][1];
		}

		long int i = 0, j = 0, k = l;
		while (i < n1 && j < n2)
		{
			if (L[i][0] <= R[j][0])
			{
				a[k][0] = L[i][0];
				a[k][1] = L[i][1];
				i++;
			}
			else
			{
				a[k][0] = R[j][0];
				a[k][1] = R[j][1];
				j++;
			}
			k++;
		}
		while (i < n1)
		{
			a[k][0] = L[i][0];
			a[k][1] = L[i][1];
			i++;
			k++;
		}
		while (j < n2)
		{
			a[k][1] = R[j][1];
			a[k][0] = R[j][0];
			j++;
			k++;
		}

		//////Merging Ends
	}
}

node *CreateNode(long int a[][2], long int m)
{
	node *n;
	n = new node;
	//n->height=1;
	n->bf = 0;
	n->C_id = a[m][0];
	n->S_price = a[m][1];
	n->left = NULL;
	n->right = NULL;
	return (n);
}

long int height(node *n)
{
	long int h;
	if (n == NULL)
		h = 0;
	else if (n->left == NULL && n->right == NULL)
	{ //n->height=1;
		h = 1;
	}
	else if (height(n->left) > height(n->right))
		h = height(n->left) + 1;
	else
		h = height(n->right) + 1;
	return (h);
}

int bf(node *n)
{
	int b;
	if (height(n) == 1)
		b = 0;
	else
		b = height(n->left) - height(n->right);
	n->bf = b;
	return b;
}

node *Anc(node *n)
{
	node *r = NULL, *q = NULL, *p = n;
	while (p != NULL)
	{
		r = q;
		q = p;
		p = p->left;
	}

	return r;
}

node *Addavl(long int a[][2], long int l, long int r)
{
	node *n;
	long int m = (r + l) / 2;
	if (l <= r)
	{
		n = CreateNode(a, m);
		n->left = Addavl(a, l, m - 1);
		n->right = Addavl(a, m + 1, r);
		/*if(n->left!=NULL && n->right!=NULL)
		  { 
		  if(n->left->height > n->right->height)
		  n->height = 1 + n->left->height;
		  else
		  n->height = 1 + n->right->height;
		  }
	  if(n->left==NULL && n->right!=NULL)
	  n->height = 1 + n->right->height;
	  if(n->left!=NULL && n->right==NULL)
	  n->height = 1 + n->left->height;
   	  if(n->left==NULL && n->right==NULL)
	  n->height=1;*/
		n->bf = bf(n);
	}
	if (l > r)
		n = NULL;

	return (n);
}

node *Insertavl(node *root, long int c_id, long int s_price)
{
	node *x = new node;
	x->bf = 0;
	x->C_id = c_id;
	x->S_price = s_price;
	x->left = NULL;
	x->right = NULL;
	node *b = NULL, *c = NULL, *q = NULL, *f = NULL, *a = NULL, *p = root;
	int d;
	if (root == NULL)
	{
		cout << c_id << " " << s_price << "\n";
		return x;
	}

	while (p != NULL)
	{
		if (bf(p) == 1 || bf(p) == -1)
		{
			f = q;
			a = p;
		}
		if (x->C_id > p->C_id)
		{
			q = p;
			p = p->right;
		}
		else
		{
			q = p;
			p = p->left;
		}
	}

	if (q->C_id < x->C_id)
		q->right = x;
	else
		q->left = x;

	if (a == NULL)
	{
		cout << c_id << " " << s_price << "\n";
		return root;
	}
	if (x->C_id < a->C_id)
	{
		b = a->left;
		d = 1;
	}
	else
	{
		b = a->right;
		d = -1;
	}

	if ((bf(a) >= -1 && bf(a) <= 1) && (bf(b) >= -1 && bf(b) <= 1))
	{
		cout << c_id << " " << s_price << "\n";
		return root;
	}
	if (d == 1)
	{
		if (bf(b) == 1)
		{ //left left rotation
			a->left = b->right;
			b->right = a;
			//linking with f
			if (f == NULL)
				root = b;
			else if (f->C_id < b->C_id)
				f->right = b;
			else
				f->left = b;
			//linked with f
			cout << c_id << " " << s_price << "\n";
			return root;
		}
		if (bf(b) == -1)
		{
			//left right rotation
			c = b->right;
			b->right = c->left;
			a->left = c->right;
			c->left = b;
			c->right = a;
			//linking with f
			if (f == NULL)
				root = c;
			else if (f->C_id < c->C_id)
				f->right = c;
			else
				f->left = c;
			//linked with f
			cout << c_id << " " << s_price << "\n";
			return root;
		}
		if (bf(b) == 0)
		{
			cout << c_id << " " << s_price << "\n";
			return root;
		}
	}
	if (d == -1)
	{
		if (bf(b) == -1)
		{
			a->right = b->left;
			b->left = a;
			//linking with f
			if (f != NULL)
			{
				if (f->C_id < b->C_id)
					f->right = b;
				else
					f->left = b;
			}
			else
				root = b;

			//linked with f
			cout << c_id << " " << s_price << "\n";
			return root;
		}
		if (bf(b) == 1)
		{
			//right left rotation
			c = b->left;
			a->right = c->left;
			b->left = c->right;
			c->left = a;
			c->right = b;
			//linking with f
			if (f == NULL)
				root = c;
			else if (f->C_id < c->C_id)
				f->right = c;
			else
				f->left = c;

			//linked with f
			cout << c_id << " " << s_price << "\n";
			return root;
		}
		if (bf(b) == 0)
		{
			cout << c_id << " " << s_price << "\n";
			return root;
		}
	}
	//cout<<c_id<<" "<<s_price<<"\n";

	return root;
}

node *Delete(node *n, long int c_id)
{ 	if (n == NULL)
		return n;
	node *f = NULL, *a = NULL, *p = n, *b = NULL, *q = NULL, *r = NULL, *ff = NULL;
	while (p != NULL && p->C_id != c_id)
	{ 	if (p->C_id < c_id)
		{
			f = a;
			a = p;
			p = p->right;
		}
		else
		{
			f = a;
			a = p;
			p = p->left;
		}
		}
	if (p == NULL)
		return n;
	//case1 : NO child
	if (p->left == NULL && p->right == NULL)
	{
		if (a == NULL)
		{
			n = NULL;
			return n;
		}
		else if (a->C_id < p->C_id)
			a->right = NULL;
		else
			a->left = NULL;
		if (bf(a) > 1) //type R
		{
			b = a->left;
			if (bf(a->left) == 0) //R0 type
			{
				a->left = b->right;
				b->right = a;
				//linking with f
				if (f == NULL)
					n = b;
				else if (f->C_id < b->C_id)
					f->right = b;
				else
					f->left = b;
				//linked with f
				return n;
			}

			if (bf(a->left) == 1) //R1 type
			{
				b->right = a;
				//linking with f
				if (f == NULL)
					n = b;
				else if (f->C_id < b->C_id)
					f->right = b;
				else
					f->left = b;
				//linked with f
				return n;
			}

			if (bf(a->left) == -1) //R -1 type
			{
				node *c = b->right;
				b->right = c->left;
				a->left = c->right;
				c->left = b;
				c->right = a;
				//linking with f
				if (f == NULL)
					n = c;
				else if (f->C_id < c->C_id)
					f->right = c;
				else
					f->left = c;
				//linked with f
				return n;
			}
		}

		if (bf(a) < -1) //type l
		{
			b = a->right;
			if (bf(a->right) == 0) //l0 type
			{
				a->right = b->left;
				b->left = a;
				//linking with f
				if (f == NULL)
					n = b;
				else if (f->C_id < b->C_id)
					f->right = b;
				else
					f->left = b;
				//linked with f
				return n;
			}

			if (bf(a->right) == -1) //l -1 type
			{
				b->left = a;
				//linking with f
				if (f == NULL)
					n = b;
				else if (f->C_id < b->C_id)
					f->right = b;
				else
					f->left = b;
				//linked with f
				return n;
			}

			if (bf(a->right) == 1) //L1 type
			{
				node *c = b->left;
				b->left = c->right;
				a->right = c->left;
				c->right = b;
				c->left = a;
				//linking with f
				if (f == NULL)
					n = c;
				else if (f->C_id < c->C_id)
					f->right = c;
				else
					f->left = c;
				//linked with f
				return n;
			}
		}
		return n;
	}
	//case 2: no anc
	if (height(p->right) == 1)
	{
		node *pp = p->right;
		pp->left = p->left;
		if (a->C_id < p->C_id)
			a->right = pp;
		else
			a->left = pp;
		return n;
	}
	if (p->right == NULL && p->left != NULL)
	{
		node *pp;
		pp = p->left;
		if (a->C_id < p->C_id)
			a->right = pp;
		else
			a->left = pp;
		return n;
	}
	if (p->left == NULL && p->right != NULL)
	{
		node *pp;
		pp = p->right;
		if (a->C_id < p->C_id)
			a->right = pp;
		else
			a->left = pp;
		return n;
	}
	//case3: With child

	r = Anc(p->right);
	//r=ff->left;
	q = r->left;
	p->C_id = q->C_id;
	p->S_price = q->S_price;
	if (height(q) == 1)
		r->left = NULL;
	else
		r->left = q->right;
	if (bf(r) == -2)
	{
		node *bb, *cc;
		ff = n;
		while (ff->left != r || ff->right != r)
		{
			if (ff->C_id < r->C_id)
				ff = ff->right;
			else
				ff = ff->left;
		}

		bb = r->right;
		r->right = bb->left;
		bb->left = r;
		if (ff->C_id < r->C_id)
			ff->right = bb;
		else
			ff->left == bb;
		return n;
	}

	return n;
}

node *UpdatePrice(node *n, long int c_id, long int s_price, long int T)
{
	node *p = n;
	while (p != NULL && p->C_id != c_id)
	{
		if (p->C_id < c_id)
			p = p->right;

		else
			p = p->left;

	}
	if (p == NULL)
		return n;

	long int k = p->S_price - s_price;
	if (k < 0)
		k = -k;
	if (k > T)
	{
		p->S_price = s_price;
		cout << p->C_id << " " << p->S_price << endl;
		return n;
	}
	else
		return n;
}

node *Stocksplit(node *n, long int c_id, double x, double y, long int T)
{
	long int s_price;

	node *p = n;
	while (p != NULL && p->C_id != c_id)
	{
		if (p->C_id < c_id)
		{
			p = p->right;
		}
		else
		{
			p = p->left;
		}
	}
	if (p == NULL)
		return n;

	s_price = int((y / x) * p->S_price);

	p->S_price = s_price;
	cout << p->C_id << " " << p->S_price << endl;
	return n;
}

int main()
{
	long int n, N, T, c_id, s_price;
	double x, y;
	char kl;
	cin >> n;
	long int a[n][2], fun;
	for (long int i = 0; i < n; i++)
		cin >> a[i][0] >> a[i][1];
	MergeSort(a, 0, n - 1);
	node *root = Addavl(a, 0, n - 1);

	cin >> N >> T;
	for (long int i = 0; i < N; i++)
	{ 	cin >> fun;

		switch (fun)
		{

		case 1:
			cin >> c_id >> s_price;
			root = Insertavl(root, c_id, s_price);
			break;
		case 2:
			cin >> c_id;
			root = Delete(root, c_id);
			break;
		case 3:
			cin >> c_id >> s_price;
			root = UpdatePrice(root, c_id, s_price, T);
			break;
		case 4:
			cin >> c_id >> x >> kl >> y;
			root = Stocksplit(root, c_id, x, y, T);
			break;
		default:
			break;
		}
	}
}
