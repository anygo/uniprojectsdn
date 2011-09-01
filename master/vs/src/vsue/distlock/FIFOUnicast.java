package vsue.distlock;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import org.jgroups.Address;
import org.jgroups.Event;
import org.jgroups.Header;
import org.jgroups.Message;
import org.jgroups.conf.ClassConfigurator;
import org.jgroups.stack.Protocol;
import org.jgroups.util.Util;

public class FIFOUnicast extends Protocol {
	private Address my_address;

	// --- Additional header for Lock request messages ---
	public static class SimulatedUnicastProtocolHeader extends Header {
		public static final short header_id = 1800;
		public Address real_destination;
		
		public Address getRealDestination() { return real_destination; }
		
		public SimulatedUnicastProtocolHeader() {}
		public SimulatedUnicastProtocolHeader(Address real_dest) {
			real_destination = real_dest;
		}
		
		@Override
		public void	readFrom(DataInputStream s) throws InstantiationException,
				IllegalAccessException, IOException {
			real_destination = Util.readAddress(s);
		}
		
		@Override
		public void	writeTo(DataOutputStream s) throws IOException {
			Util.writeAddress(real_destination, s);
		}
		
		@Override
		public int size() {
			return 0;
		}
	}
	
	@Override
	public Object down(Event evt) {
		switch (evt.getType()) {
		case Event.MSG:
			Message m = (Message)evt.getArg();
			if (m.getDest() != null) {
				SimulatedUnicastProtocolHeader ph = 
						new SimulatedUnicastProtocolHeader(m.getDest());
				m.setDest(null);
				m.putHeader(SimulatedUnicastProtocolHeader.header_id, ph);
			}
			break;
		}
		return down_prot.down(evt);
	}
	
	@Override
	public Object up(Event evt) {
		switch (evt.getType()) {
		case Event.MSG:
			Message m = (Message)evt.getArg();
			SimulatedUnicastProtocolHeader ph = (SimulatedUnicastProtocolHeader)
				m.getHeader(SimulatedUnicastProtocolHeader.header_id);
			if (ph != null) {
				m.setDest(ph.getRealDestination());
				if (! ph.getRealDestination().equals(my_address)) return null;
			}
			break;
		case Event.VIEW_CHANGE:
			my_address = getProtocolStack().getChannel().getAddress();
			break;
		}
		return up_prot.up(evt);
	}
	
	@Override
	public void init() {
		ClassConfigurator.add(
				SimulatedUnicastProtocolHeader.header_id,
				SimulatedUnicastProtocolHeader.class);
	}
}
