package mw;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;


public class MWClientThreadPool {
	
	public MWClientThreadPool(MWClient client, String query, int poolsize, int n) {
		
		ExecutorService exec = Executors.newFixedThreadPool(poolsize);
		AtomicLong counter = new AtomicLong(0);
		AtomicLong gesamtZeit = new AtomicLong();
		
		
		for(int i = 0; i < poolsize; i++) {
			exec.execute(new Worker(client, n, query, counter, gesamtZeit));
		}
		exec.shutdown();
		while(!exec.isTerminated()) {
			try {
				Thread.sleep(2000);
				System.out.println("Durchschnittliche Zeit pro Anfrage: " + (double)gesamtZeit.get()/(double)counter.get() + "ms");
				counter.set(0);
				gesamtZeit.set(0);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

	}
	
	private class Worker implements Runnable {
		
		private MWClient client;
		private int n;
		private String query;
		private AtomicLong counter;
		private AtomicLong gesamtZeit;
		
		
		public Worker(MWClient client, int n, String query, AtomicLong counter, AtomicLong gesamtZeit) {
			
			this.client = client;
			this.n = n;
			this.query = query;
			this.counter = counter;
			this.gesamtZeit = gesamtZeit;
		}
		
		public void run() {
			
			for(int i = 0; i < n; i++) {
				long start = System.currentTimeMillis();
				client.searchIDs(query);
				long took = System.currentTimeMillis() - start;
				gesamtZeit.addAndGet(took);
				counter.addAndGet(1);
				
			}
		}
		
	} 
		
		


}
