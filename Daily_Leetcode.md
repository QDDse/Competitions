

#                                                                                                           算法

[TOC]

## ==python==

### 1. iterator && iterable

- 可迭代的对象--> iterable

  - list、 dict、 tuple (**可以放在for loop中的都是iterable**) 

  - 含有` __iter__()` 方法并且`iter()` 返回的是一个包含`next（）`方法的迭代器

    ~~~python
    ## 可以放在for loop
    for i in list:
        pass
    for i in dict:
        pass
    ~~~

-  iterator ----> 迭代器

  - class 中包含有``__next__()` 方法
  - iterator 可以直接调用`next()`函数 一个一个返回containor中的数据。
  - list、dict、tuple这些是iterable 而不是 iterator

------



## 1. Daily Leetcode



## *Optimizer（优化）*

### 1. carry = sum>10?1:0;

~~~java
carry = sum/10;
//更改为 
carry = sum>10?1:0;  速度改进（up）
~~~

 ## 2. reverse

- ```
  nums = "----->-->"; k =3
  result = "-->----->";
  
  reverse "----->-->" we can get "<--<-----"
  reverse "<--" we can get "--><-----"
  reverse "<-----" we can get "-->----->"
  this visualization help me figure it out :)
  ```





## 2. Graph

### 2.1 图的两种表示形式

- 邻接表
- 邻接矩阵
- ![img](E:/OneDrive/%E6%96%87%E6%A1%A3/Notes/images/2-16473267086562.jpeg)

~~~ java
//邻接表 占用空间小，但是查询x是否是y的邻点复杂
// graph[x] 存储x的邻接点
List<Integer>[] graph;  

// 邻接矩阵，矩阵空白元素浪费空间，但是查找高效
boolean[][] matrix;
~~~

### 2.2 graph的遍历

~~~ java
// 因为图类似多叉树的遍历，但是具有环，需要visited[]
boolean[] visited;
//记录从start到当前节点的path
boolean[] path;

void traverse(Graph graph, int s){
    if(visited[s]) return;
    visited[s] = true;
    path[s] = true;
    for(int neighbor : graph.neighbors(s)){
        traverse(graph, neighbor);
    }
    // Undo 到当前节点的path
    path[s] =
}
~~~

​	

​	

## 3. Tree

> - `Tree`算法主要依赖于遍历。 
> - 可以利用`栈` 进行树的迭代遍历
> - 重要性质：
>   - n 个顶点， n-1的边
>   - 任何一个节点到根节点存在`唯一`路径，路径的长度为节点所处的深度

> `Binary Tree`（二叉树）：
>
> - 相关算法题：
>   - 94,
>   - 102
>   - 103
>   - 144
>   - 145
>   - 199
> - `堆`： 本质就是二叉树 ， 一种优先队列`Priority queue`
>   - 295
> - `Binary Sort Tree`（二叉查找树）：
>   - 中序遍历是有序数列
>   - 98
> - `二叉平衡树` 
> - `红黑树`： 有良好的最坏运行情况，可以在O(logn)内完成查找，insert，delete

### 3.1 Trie（字典树）

> - 根节点不包含字符， 除root以外的node都只包含一个字符
> - 每个节点的所有子节点包含的字符都不相同

~~~c++
class Trie{
private:
    bool isEnd;
    Trie* next[26];
public:
    // 构造函数
    Trie(){
        inEnd = false;
        memset(next, 0, sizeof(next));
    }
    // 插入函数
    void Insert(string word){
        Trie* node = this;
        for(char c : word){
            if(node->next[c-'a'] == NULL){
                node->next[c-'a'] = ner Trie();
            }
            node = node->next[c-'a'];
        }
        node->isEnd = true;
    }
    bool Search(string word){
        Trie* node = this;
        for(char c : word){
            node = node->next[c - 'a'];
            if(node == NULL)
                return fasle;
        }
        return node->isEnd;
    }
    bool StartWith(string prefix){
        Trie* node = this;
        for(char c : prefix){
            node = node->next[c - 'a'];
            if(node == NULL)
                return false;
        }
        return true;
    }
}
~~~

### 3.2 从二叉堆——> 优先级队列

#### 1. 二叉堆

~~~c++
class MaxPQ{
public:
    //存储元素的数组
    vector<int> pq;
    int N;   // num
    
}
~~~

#### 2. 直接使用priority_queue

~~~c++
priority_queue<int, vector<int> > pq;
// 方法：
pq.top();
pq.insert();
pq.empty();
~~~

~~~c++
// 将k个升序列表合并
class Solution{
public:
    ListNode * KListMerge(vector<ListNode *> lists){
        //将所有元素放进pq
        //默认为大顶堆
        priority_queue<int> pq;
        for(const auto& list : lists){
            while(list){
                pq.insert(list->val);
                list = list->next;
            }
        }
        ListNode * res;
        while(!pq.empyt()){
            ListNode * p =  new ListNode(pq.top());
            p->next = res;
            res = p;
        }
    }
    return res;
}
~~~

### 3.3 Tree 的序列化

- 序列化---- 持久化

**LeetCode#652**------------ 寻找重复子树

- 用到haspmap in c++
  - unordered_map<type, type>

~~~c++
class Solution{
public:
    string Getnode(vector<TreeNode*> &res, unordered_map<string, int> &map, TreeNode *root){
        //base case 
        if(root == nullptr){
            return "";
        }
        string tmp = to_string(root->val) + "," + Getnode(res, map, root->left) + "," + Getnode(res, map, root->right);
        //map 中tmp对应的int 为1 则表示已经有该序列，root加入res中
        if(map[tmp] == 1){
            res.push_back(root);
        }
        // 否则将tmp加入map中并int set=1
        else{
            map[tmp]++;
        }
        return tmp;
    }
    vector<TreeNode*> answer(TreeNode* root){
        //初始化res，map
        vector<TreeNode*> res;
        unordered_map<string, int> map;
        Getnode(res, map, root);
        return res;
    }
}
~~~







## 4. ListNode 链表

### 4.1 获取单链表倒数k个

> - 一原则： 画图
> - 二考点： 指针的修改； 链表的拼接
> - 三注意： 
>   - 出现环，造成死循环 （判断是否有环， 环的位置）
>   - 边界出错
>   - *递归*:  多数listnode的 题是单链表，只有前序后序遍历
> - 四个方法：
>   - 虚拟头结点
>   - 快慢指针
>   - 穿针引线
>   - 

- 分析： 双指针，p1从head 遍历k ， 然后p2从head开始与p1同时向后遍历，则p2 遍历了n-k 即倒数第k个

~~~c++
~~~

### 4.2 链表递归

- 反转链表

~~~c++
ListNode reverse(ListNode * head){
    //base case 
    if(head == nullptr){
        return NULL;
    }
    if(head->next == nullptr){
        return head;
    }
    ListNode * last = reverse(head->next);
    head->next->next = head;
    head->next = nullptr;
    return last;
}
~~~

- 将链表的n-->m位置反转

~~~c++
ListNode reverseBetween(ListNode head, int n, int m){
    // base case 
    if(m == 1){
        return reverseN(head, n);
    }
    head->next = reverseBetween(head->next, m-1, n-1);
    return head;
}
//反转head开头的n个链表
ListNode reverseN(ListNode* head, int n){
    ListNode* successor = nullptr;
    // base case 
    if(n == 1){
        return head;
        successor = head->next;
    }
    ListNode * last = reverseN(head->next, n-1);
    head->next->next = head;
    head->next = successor;
    return last; // last 成为了head;
}
~~~

## 5. 逆序对

- 数字中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。
- ![image-20220330165815565](E:/OneDrive/%E6%96%87%E6%A1%A3/Notes/images/image-20220330165815565-16486306969521.png)





## 6.DP

> - 关键是找到相应的`dp`数组的定义
> - 假设`dp`数组的[0, n-1]都知道 推出dp[n]

- 一种穷举策略，同时具有memo table（备忘录）

### 6.1 斐波那契函数

- ~~~c++
  int fib(int n){
  	vector<int> dp(N + 1, 0)   //vector作为容器 <> 为数据类型, N+1为最大容量， 0为每个对象的初始值
      dp[1] = dp[2] = 1;
      for(int i = 3;i<=n;i++){
          dp[i] = dp[i-1]+dp[i-2];
      }
      return dp[n]
  }
  ~~~

- 

### 6.2 最长递增子序列（Longest Increasing Subsequence）

- `子序列`------<font color=red> 可以在列表中不连续</font>
- `子串` ----- <font color=red> 一定是连续的</font>

~~~c++
class Solution{
public:
    int lengthodLIS(vectot<int> & nums){
        // 定义dp数组为以nums[i]为tail的最长LIS
        vector<int> dp(nums.size(), 1); //用1进行初始化
        for(int i = 0; i < nums.size(); i++){
            for(int j = 0; j < i; j++){
                if(nums[i] > nums[j])
                    dp[i] = max(dp[i], dp[j]+1);
            }
        }
        auto maxposition = max_element(dp.begin(), dp.end());
        return dp[maxposition - dp.begin()];
    }
};
~~~

### 6.3 俄罗斯套娃信封-----二维数组的最长递归子序列

![img](E:/OneDrive/%E6%96%87%E6%A1%A3/Notes/images/title-16501215297142.png)

- <font size=4, color='blue'> **先将w进行升序， 再对w相同的pair的h降序，之后对h查找LIS即可**</font>

![img](E:/OneDrive/%E6%96%87%E6%A1%A3/Notes/images/2-16501221267654.jpg)

~~~cpp
class Solution {
public:
    int maxEnvelopes(vector<vector<int>>& envelopes) {
        //对w记性sort
        int n = envelopes.size();
        sort(envelopes.begin(), envelopes.end(), [](const auto& e1, const auto& e2){
            return e1[0] < e2[0] || ((e1[0] == e2[0]) && (e1[1] >= e2[1]));
        });
        vector<int> dp {envelopes[0][1]};
        for(int i = 0; i < n; i++){
            int num = envelopes[i][1];
            if(num > dp.back()){
                dp.push_back(num);
            }
            else{
                auto it = lower_bound(dp.begin(), dp.end(), num); //找到大于等于num的第一个数对应的 位置
                *it = num; //对it位置value更新为num
            }
        }
        return dp.size();
    }
};
~~~

- <font size=3, color=red>`sort`函数</font>

  - ~~~cpp
    sort(RandomAccessIterator first, RandomAccessIterator last, Compare cmp);
    //cmp函数可以重写
    bool cmp(const auto& m1, const auto& m2){
        return m1 > m2
    }//逆序排序
    //用lambda 函数
    sort(vectot.begin(), vector.end(), [](const auto& m1, const auto& m2){
        return m1.weight < m2.weight;
    });
    ~~~

- <font size=3, color=red>`lower_bound`函数</font>

- ~~~cpp
  // 找到不小于num的第一个数的位置
  auto it = lower_bound(start, end, num)
  ~~~

### 6.4 最长subArray

~~~cpp
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        //建立dp数组--- 以nums[i]为结尾的最大subArray
        vector<int> dp(nums.size());
        if(nums.size() == 0)    return 0;
        // 第一个元素自成一派
        dp[0] = nums[0];
        //由dp[i-1] ---> dp[i]
        for(int i = 1; i < nums.size(); i++){
            dp[i] = max(dp[i-1] + nums[i], nums[i]);
        }
        // int 最小值
        int res = INT_MIN;
        for(int i = 0; i < dp.size(); i++){
            res = max(res, dp[i]);
        }
        return res;
    }
};
//优化内存    整个过程只跟dp[i-1]有关
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        //建立dp数组--- 以nums[i]为结尾的最大subArray
        int dp0 = nums[0];
        int dp1 = 0;
        if(nums.size() == 0)    return 0;
        // 第一个元素自成一派
        //由dp[i-1] ---> dp[i]
        int res = dp0;
        for(int i = 1; i < nums.size(); i++){
            dp1 = max(dp0 + nums[i], nums[i]);
            res = max(dp1, res);
            dp0 = dp1;
        }

        return res;
    }
};
~~~

 

### 6.5 最长公共子序列（LCS） 

[^双字符串DP]: 经典的双字符串Dp之一

### 6.6 编辑距离



![img](E:/OneDrive/%E6%96%87%E6%A1%A3/Notes/images/title-16506810154531-16506810174723.png)

- **解决methods**

![img](E:/OneDrive/%E6%96%87%E6%A1%A3/Notes/images/dp-16509035629142.jpg)

~~~cpp
class Solution{
public:
    int get_min(int a, int b, int c){
        return min(a, min(b, c));
    }
    int minDistance(string word1, string word2){
        int m = word1.size(), n = word2.size();
        vector<vector<int> dp(m+1, vector<int>(n+1));
        //base case (见上图)
        for(int i = 0; i <= m; i++){
            dp[i][0] = i;
        }
        for(int i = 0; i <= n; i++){
            dp[0][i] = i;
        }
        for(int i = 1; i <= m; i++){
            for(int j = 1; j <= n; j++){
                if(word1[i-1] == word2[j]){
                    dp[i][j] = dp[i-1][j-1];
                }
                else
                    dp[i][j] = get_min(dp[i-1][j], 
                                      dp[i][j-1],
                                      dp[i-1][j-1]
                                      )
            }
        }
        return dp[m][n];
    }
}
~~~





### 6.7 KMP

~~~cpp
class Solution {
public:
    int strStr(string haystack, string needle) {
        int n = haystack.size(), m = needle.size();
        if(m == 0)  return 0;
        //设置哨兵
        haystack.insert(haystack.begin(), ' ');  //insert函数在begin位置插入''
        needle.insert(needle.begin(), ' ');
        vector<int> next(m+1);
        //预处理next
        for(int i = 2, j = 0; i <= m; i++){
            while(j and needle[i] != needle[j+1])
                j = next[j];
            if(needle[i] == needle[j+1])
                j++;
            next[i] = j;
        }
        //匹配过程
        for(int i = 1, j =0; i <= n; i++){
            while(j and haystack[i] != needle[j+1])
                j = next[j];
            if(haystack[i] == needle[j+1])  
                j++;
            if(j == m)
                return i - m;
        }
        return -1;
    }
};
~~~



------



# **CPP**

## 变量

- C++中变量名次 也叫`标识符`, <font color=orange>只能用字母、数字和下划线组成</font >
- `数字`不能作为开头
- 变量的初始化需要分别进行

## 常量

- `字面量` ---

## Data Structure

### 1. Priority Queue

> - `top`: 访问队头元素
> - `empty`: 队列是否空
> - `size`: return 队列中元素个数
> - `push`:插入元素到队尾（并排序）
> - `emplace`:原地构建一个元素并插入队列
> - `pop`: 弹出队头元素
> - `swap`: 交换内容

~~~cpp
#include <iostream>
#include <queue>
#include <vector>
using namespace std;
int main(){
    priority_queue<pair<int, int>> a;
    priority_queue<pair<int ,int>, less<int>> b; // 大顶堆
    priority_queue<pair<int, int>, greater<int>> c; //小顶堆
    
}
~~~



### 2. 链表

> <font size=4, color=red> 各种数据结构底层本质都是数组和链表</font>
>
> 

