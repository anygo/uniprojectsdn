package vsue.distlock;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import org.jgroups.Event;
import org.jgroups.Global;
import org.jgroups.Header;
import org.jgroups.Message;
import org.jgroups.conf.ClassConfigurator;
import org.jgroups.stack.Protocol;

public class VSLogicalClockProtocol extends Protocol {
	private Object locker = new Object();
	private int counter = 0;

	public static class ClockHeader extends Header {
		public static final short header_id = 1501;

		private int counter;

		public int getCounter() {
			return counter;
		}

		public void setCounter(int counter) {
			this.counter = counter;
		}

		@Override
		public void writeTo(DataOutputStream s) throws IOException {
			s.writeInt(counter);
		}

		@Override
		public void readFrom(DataInputStream s) throws IOException {
			this.counter = s.readInt();
		}

		@Override
		public int size() {
			return Global.BYTE_SIZE + Integer.SIZE;
		}
	}

	public static int getMessageTime(Message m) {
		return ((ClockHeader) m.getHeader(ClockHeader.header_id)).getCounter();
	}

	@Override
	public void init() {
		ClassConfigurator.add(ClockHeader.header_id, ClockHeader.class);
	}

	@Override
	public Object down(Event evt) {
		if (evt.getType() == Event.MSG) {
			Message m = (Message) evt.getArg();
			ClockHeader clkHdr = new ClockHeader();

			synchronized (locker) {
				++counter;
				clkHdr.setCounter(counter);

				m.putHeader(ClockHeader.header_id, clkHdr);
				evt = new Event(Event.MSG, m);
				return down_prot.down(evt);
			}
		}
		return down_prot.down(evt);
	}

	@Override
	public Object up(Event evt) {
		if (evt.getType() == Event.MSG) {
			Message m = (Message) evt.getArg();
			int msgTime = getMessageTime(m);

			synchronized (locker) {
				counter = (msgTime > counter) ? msgTime + 1 : counter + 1;
			}
		} else {
			synchronized (locker) {
				++counter; // TODO: Richtig?
			}
		}
		return up_prot.up(evt);
	}
}