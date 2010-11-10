
class TreeNode {
	TreeNode left;
	TreeNode right;
	int key;

	public TreeNode(int new_key) {
		key = new_key;
		left = null;
		right = null;
	}

	public int nodeCount() {
		if (left == null && right == null) return 1;
		if (left == null && right != null) return (right.nodeCount() + 1);
		if (left != null && right == null) return (left.nodeCount() + 1);
		else return (left.nodeCount() + right.nodeCount() + 1);
	}

	public String toString() {
		if (left == null && right == null) return key + " ";
		if (left == null && right != null) return key + " " + right.toString();
		if (left != null && right == null) return left.toString() + key + " ";
		else return left.toString() + key + " " + right.toString();
	}
}

public class Tree {	
	TreeNode root;

	public void add(int a) {
		if (root == null) {
			root = new TreeNode(a);
		}
		TreeNode tmp = root;
		TreeNode tmpReminder = null;
		while (tmp != null) {
			tmpReminder = tmp;
			if (a == tmp.key) return;
			if (a < tmp.key) {
				tmp = tmp.left;
			} else {
				tmp = tmp.right;
			}
		}
		tmp = new TreeNode(a);

		if (tmpReminder != null) {
			if (tmp.key < tmpReminder.key) {
				tmpReminder.left = tmp;
			} else {
				tmpReminder.right = tmp;
			}
		}
	}

	public boolean contains(int a) {
		TreeNode tmp = root;
		while (tmp != null) {
			if (a == tmp.key) return true;
			if (a < tmp.key) {
				tmp = tmp.left;
			} else {
				tmp = tmp.right;
			}
		}
		return false;
	}

	public int count() {
		if (root == null) return 0;
		else return root.nodeCount();
	}

	public String toString() {
		if (root == null) return "";
		else return root.toString();
	}
}
