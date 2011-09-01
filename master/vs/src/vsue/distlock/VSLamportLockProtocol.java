package vsue.distlock;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.TreeMap;
import java.util.concurrent.Semaphore;

import org.jgroups.Address;
import org.jgroups.Event;
import org.jgroups.Global;
import org.jgroups.Header;
import org.jgroups.Message;
import org.jgroups.View;
import org.jgroups.conf.ClassConfigurator;
import org.jgroups.stack.Protocol;

public final class VSLamportLockProtocol extends Protocol {
	public static final byte RELEASE = 2;
	public static final byte REQUEST = 4;
	public static final byte ACK = 8;

	private Object locker = new Object();
	private Address addr;
	private View view;
	private TreeMapKey ourLastRequest;
	private Semaphore waitUntilLockAcquiredSemaphore;
	private byte nextMessageType = 0;
	private TreeMap<TreeMapKey, Integer> sperranfragen = new TreeMap<TreeMapKey, Integer>();

	public static class LockProtocolHeader extends Header {
		public static final short header_id = 1500;
		private byte type;

		@Override
		public void readFrom(DataInputStream s) throws IOException {
			this.type = s.readByte();
		}

		@Override
		public void writeTo(DataOutputStream s) throws IOException {
			s.writeByte(this.type);
		}

		@Override
		public int size() {
			return Global.BYTE_SIZE + Byte.SIZE;
		}

		public byte getType() {
			return type;
		}

		public void setType(byte type) {
			this.type = type;
		}
	}

	public void setWaitUntilLockAcquiredSemaphore(Semaphore semaphore) {
		this.waitUntilLockAcquiredSemaphore = semaphore;
	}

	public void setNextMessageType(byte type) {
		this.nextMessageType = type;
	}

	@Override
	public void init() {
		ClassConfigurator.add(LockProtocolHeader.header_id, LockProtocolHeader.class);
	}

	@Override
	public Object down(Event evt) {
		switch (evt.getType()) {

		case Event.MSG:
			Message msg = (Message) evt.getArg();

			if (nextMessageType == REQUEST) {
				LockProtocolHeader hdr = new LockProtocolHeader();
				hdr.setType(VSLamportLockProtocol.REQUEST);
				msg.putHeader(LockProtocolHeader.header_id, hdr);
				nextMessageType = 0;
			} else if (nextMessageType == RELEASE) {
				LockProtocolHeader hdr = new LockProtocolHeader();
				hdr.setType(VSLamportLockProtocol.RELEASE);
				msg.putHeader(LockProtocolHeader.header_id, hdr);
				nextMessageType = 0;
			}
			break;
		case Event.SET_LOCAL_ADDRESS:
			Address newAddr = (Address) evt.getArg();
			synchronized (locker) {
				addr = newAddr;
			}
			break;
		case Event.VIEW_CHANGE:
			View newView = (View) evt.getArg();
			synchronized (locker) {
				view = newView;
			}
			break;
		}
		return down_prot.down(evt);
	}

	@Override
	public Object up(Event evt) {
		switch (evt.getType()) {
		
		case Event.MSG:
			Message m = (Message) evt.getArg();
			LockProtocolHeader hdr;

			if ((hdr = (LockProtocolHeader) m.getHeader(LockProtocolHeader.header_id)) != null) {
				switch (hdr.getType()) {
				
				case VSLamportLockProtocol.REQUEST:
					// send ACK
					Message ack = new Message(m.getSrc(), addr, m.getRawBuffer(), m.getOffset(), m.getLength());
					LockProtocolHeader h = new LockProtocolHeader();
					h.setType(VSLamportLockProtocol.ACK);
					ack.putHeader(LockProtocolHeader.header_id, h);
					Event ackEvent = new Event(Event.MSG, ack);

					// put in my list
					TreeMapKey tmk = new TreeMapKey(VSLogicalClockProtocol.getMessageTime(m), m.getSrc());
					synchronized (locker) {
						sperranfragen.put(tmk, 0);
						if (addr.equals(m.getSrc())) {
							ourLastRequest = tmk;
						}
					}

					down_prot.down(ackEvent);
					break;
					
				case VSLamportLockProtocol.RELEASE:
					synchronized (locker) {
						sperranfragen.remove(sperranfragen.firstKey());
						grantLockAccess();
					}
					break;
					
				case VSLamportLockProtocol.ACK:
					synchronized (locker) {
						sperranfragen.put(ourLastRequest, sperranfragen.get(ourLastRequest) + 1);
						grantLockAccess();
					}
					break;
				}
				
				return null;
			}
			break;
			
		case Event.SET_LOCAL_ADDRESS:
			Address newAddr = (Address) evt.getArg();
			synchronized (locker) {
				addr = newAddr;
			}
			break;
			
		case Event.VIEW_CHANGE:
			View newView = (View) evt.getArg();
			synchronized (locker) {
				view = newView;
			}
			break;
		}
		
		return up_prot.up(evt);
	}

	private void grantLockAccess() {
		if (!sperranfragen.containsKey(ourLastRequest)) {
			return;
		}

		TreeMapKey firstKeyInList = sperranfragen.firstKey();
		if (firstKeyInList.compareTo(ourLastRequest) == 0 && sperranfragen.get(firstKeyInList) == view.getMembers().size()) {
			waitUntilLockAcquiredSemaphore.release();
		}
	}
}
