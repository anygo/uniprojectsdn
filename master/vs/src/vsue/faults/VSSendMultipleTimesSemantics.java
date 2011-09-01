package vsue.faults;

import java.lang.reflect.Method;
import java.rmi.RemoteException;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Semaphore;

import vsue.VSConstants;
import vsue.communication.VSMessage;
import vsue.communication.VSMessageReply;
import vsue.replica.VSRemoteGroupReference;

public class VSSendMultipleTimesSemantics extends VSAbstractRPCSemantics {
	
	private VSMessageReply reply;
	private Semaphore semaphore;
	private CountDownLatch latch;
	private VSMessage message;
	private int counter;
	
	public VSSendMultipleTimesSemantics(VSRemoteGroupReference groupReference) {
		super(groupReference);
		
		this.semaphore = new Semaphore(1);
		counter = 0;
	}

	@Override
	public Object invoke(Object proxy, Method method, Object[] args)
			throws Throwable {
		Timer timer = null;
		TimerTask task = null;
		counter = 0;

		message = generateMessage(method, args);
		
		latch = new CountDownLatch(1);
		timer = new Timer();
		task = new VSTimeoutHandler();
		timer.schedule(task, 0);
		
		latch.await();
		
		// falls z.b. der zweite thread vorm ersten fertig ist, muss der erste hier gecancelt werden
		semaphore.acquire();
		task.cancel();
		timer.cancel();
		semaphore.release();

		if (reply.getReturnValue() instanceof Throwable) {
			throw (Throwable) reply.getReturnValue();
		}
		return reply.getReturnValue();
	}

	private void sendAndStartTimer() throws Throwable {
		Timer timer = null;
		TimerTask task = null;
		VSMessageReply r = null;
		VSMessage clone = null;

		semaphore.acquire();
		// zu viele versuche? abbruch
		if (counter > VSConstants.SEND_MULTIPLE_TIME_MAX_ATTEMPTS) {
			reply = new VSMessageReply(0, 0, new RemoteException(
					"SEND_MULTIPLE_TIME_MAX_ATTEMPTS"));
			latch.countDown();
			semaphore.release();
			return;
		} else {
			counter++;
		}
		semaphore.release();

		// zuerst neuen task starten
		timer = new Timer();
		task = new VSTimeoutHandler();
		timer.schedule(task, VSConstants.SEND_MULTIPLE_TIME_WAITING_TIME);
		
		// clone darf waehrend dem Senden nicht veraendert werden koennen
		semaphore.acquire();
		clone = new VSMessage(message);
		semaphore.release();
		r = sendAndReceive(clone);
		
		semaphore.acquire();
		if (r.getRemoteCallID() == message.getRemoteCallID()
				&& r.getSequenceNumber() == message.getSequenceNumber()) {
			System.out.println("Antwort passt zu gesendeter Nachricht");
			// rechtzeitig fertig... also oben gestarteten task abbrechen
			task.cancel();
			timer.cancel();
			reply = r;
			latch.countDown();
		} else {
			System.out.println("Antwort passt nicht zu gesendeter Nachricht");
		}
		semaphore.release();
	}

	private class VSTimeoutHandler extends TimerTask {

		public VSTimeoutHandler() {
		}

		public void run() {
			try {
				semaphore.acquire();
				message.incrementSequenceNumber();
				semaphore.release();

				sendAndStartTimer();
			} catch (Throwable e) {
				e.printStackTrace();
			}
		}
	}

}
