package vsue.distlock;

import java.util.concurrent.Semaphore;
import org.jgroups.JChannel;
import org.jgroups.Message;

public class VSLamportLock {
	private JChannel channel;
	private VSLamportLockProtocol lamportProtocol;

	public VSLamportLock(JChannel channel) {
		this.channel = channel;
		this.lamportProtocol = (VSLamportLockProtocol) channel.getProtocolStack().findProtocol(VSLamportLockProtocol.class);
	}

	public void lock() throws Exception {
		Semaphore semaphore = null;
		Message msg = null;

		semaphore = new Semaphore(0);
		msg = new Message(null, channel.getAddress(), "Lock");
		lamportProtocol.setNextMessageType(VSLamportLockProtocol.REQUEST);
		lamportProtocol.setWaitUntilLockAcquiredSemaphore(semaphore);
		channel.send(msg);

		semaphore.acquire();
	}

	public void unlock() throws Exception {
		Message msg = null;

		msg = new Message(null, channel.getAddress(), "Unlock");
		lamportProtocol.setNextMessageType(VSLamportLockProtocol.RELEASE);
		channel.send(msg);
	}
}
