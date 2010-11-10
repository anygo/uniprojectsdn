
public class Heap {
	
	private int[] heap;
	private int last;

	public Heap(int capacity) {
		heap = new int[capacity];
		last = 0;
	}

	public int left(int idx) {
		if (heap[2*idx+1] == null)
			return -1;
		return 2*idx+1;
	}

	public int right(int idx) {
		if (heap[2*idx+2] == null)
			return -1;
		return 2*idx+2;
	}

	public int parent(int idx) {
		if (idx/2-1 < 0)
			return -1;
		return idx/2-1;
	}

	public void add(int a) {
		if (heap[0] == null) {
			heap[0] = a;
			return;
		}
		heap[last] = a;
		int idx = last++;

		while (parent(idx) != -1) {
			if (heap[parent(idx)] > a)
				return;
			swap(parent(idx), idx);
			idx = parent(idx);
		}
	}

	public void pop() {
		last--;
		heap[0] = heap[last];
		heap[last] = null;
		int idx = 0;
		while (left(idx) != -1) {
			if (heap[idx] > heap[left(idx)] {
				return;
			}
			if (right(idx) != -1) {
				if (heap[right(idx)] > heap[left(idx)]) {
					swap(idx, right(idx));
				} else swap(idx, left(idx));
			}
			if (heap[left(idx)] > heap[idx]) {
				swap(idx, left(idx);
			}
			idx =
		}
	}



	public void swap(int i1, int i2) {
		int tmp = heap[i1];
		heap[i1] = heap[i2];
		heap[i2] = tmp;
	}
}
