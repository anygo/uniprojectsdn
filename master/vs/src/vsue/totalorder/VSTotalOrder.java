package vsue.totalorder;

import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import org.jgroups.*;
import org.jgroups.conf.ClassConfigurator;
import org.jgroups.stack.Protocol;

import vsue.VSConstants;

public class VSTotalOrder extends Protocol {

	private Address addr;
	private View view;
	private boolean leaderFlag;
	private Address leaderAddr;
	private long no = 0;
	private Object locker = new Object();

	// aufgabe 5.3 und 5.4 - Verwaltungsstrukturen
	private Map<VSMsgID, Message> msgBuffer;
	private Map<VSMsgID, Integer> msgACKCounter;
	private List<VSMsgID> orderedMessageQueue;

	public VSTotalOrder() {
		msgBuffer = new HashMap<VSMsgID, Message>();
		msgACKCounter = new HashMap<VSMsgID, Integer>();
		orderedMessageQueue = new LinkedList<VSMsgID>();
	}

	@Override
	public void init() throws Exception {
		super.init();

		ClassConfigurator.add(id, VSTotalOrderHeader.class);
	}

	@Override
	public Object down(Event evt) {
		switch (evt.getType()) {

		case Event.MSG: {
			Message msg = (Message) evt.getArg();
			// ueberpruefen ob Multicast
			if (!(msg.isFlagSet(Message.NO_TOTAL_ORDER) || msg.getDest() != null
					&& !msg.getDest().isMulticastAddress())) {

				if (msg.getSrc() == null)
					msg.setSrc(addr);

				// aufgabe 5.4
				if (VSConstants.OPTIMIZED_ACK_STRATEGY) {
					Message newMsg = new Message(null, addr,
							msg.getRawBuffer(), msg.getOffset(),
							msg.getLength()); // Nachricht mit Nutzdaten an
					VSMsgID vsmsgid;
					synchronized (locker) {
						vsmsgid = new VSMsgID(addr, ++no);
					}
					newMsg.putHeader(id,
							VSTotalOrderHeader.createMulticast(vsmsgid));

					Event modifiedEvt = new Event(Event.MSG, newMsg);
					return super.down(modifiedEvt);
				} else {
					Message newMsg = new Message(leaderAddr, addr,
							msg.getRawBuffer(), msg.getOffset(),
							msg.getLength()); // Nachricht mit Nutzdaten an
												// Leader
					VSMsgID vsmsgid;
					synchronized (locker) {
						vsmsgid = new VSMsgID(addr, ++no);
					}
					newMsg.putHeader(id,
							VSTotalOrderHeader.createRerouting(vsmsgid));

					Event modifiedEvt = new Event(Event.MSG, newMsg);
					return super.down(modifiedEvt);
				}

			} else {
				return super.down(evt);
			}
		}

		case Event.SET_LOCAL_ADDRESS: {
			Address newAddr = (Address) evt.getArg();
			synchronized (locker) {
				addr = newAddr;
			}
			return super.down(evt);
		}

		case Event.VIEW_CHANGE: {
			View newView = (View) evt.getArg();
			synchronized (locker) {
				view = newView;
				if (!view.getMembers().isEmpty()) {
					leaderAddr = view.getMembers().firstElement();
					leaderFlag = (leaderAddr.equals(addr));
				} else {
					System.err.println("view.getMembers(): leere Liste.");
				}
			}
			return super.down(evt);
		}

		default: {
			return super.down(evt);
		}

		}
	}

	private Object iterateOrderedMessageQueue() {
		Object ret = null;

		List<Event> events = new LinkedList<Event>();

		synchronized (locker) {

			//System.out.println("iterateOrderedMessageQueue(), size = "
				//	+ orderedMessageQueue.size());

			Iterator<VSMsgID> it = orderedMessageQueue.iterator();
			while (it.hasNext()) {
				VSMsgID curMsgID = it.next();
				if (view != null && view.getMembers() !=null && msgACKCounter.get(curMsgID) >= view.getMembers().size()) {

					if (!msgBuffer.containsKey(curMsgID)) {
						//System.out.println("Nachricht noch nicht da");
						break;
					}
					Message curMsg = msgBuffer.get(curMsgID);

					Event newEvt = new Event(Event.MSG, curMsg);

					//System.out.println("genug ACKs, Nachricht nach oben: "
							//+ curMsg);
					// aus liste entfernen
					it.remove();

					events.add(newEvt);
				} else {
					///System.out.println("noch nicht genug ACKs "
							//+ msgACKCounter.get(curMsgID) + " / "
							//+ view.getMembers().size());
					break; // auf richtige Reihenfolge achten, nur die nach oben
							// geben, die auch dran sind
				}
			}
		}

		for (Event e : events) {
			ret = super.up(e);
		}

		return ret;
	}

	@Override
	public Object up(Event evt) {
		switch (evt.getType()) {

		case Event.MSG: {
			// Process incoming messages
			Message msg = (Message) evt.getArg();
			VSTotalOrderHeader hdr = (VSTotalOrderHeader) msg.getHeader(id);
			if (hdr == null) {
				return super.up(evt);
			} else if (hdr.getMsgType() == VSTotalOrderMsgType.MULTICAST) {

				if (VSConstants.OPTIMIZED_ACK_STRATEGY) {
					System.out
							.println("isOrdering: " + hdr.isOrderingMessage() + " from: " + msg.getSrc());
					if (hdr.isOrderingMessage()) {
						synchronized (locker) {
							orderedMessageQueue.add(hdr.getMsgID());

							//System.out
									//.println("Ordering Message ist angekommen, versende ACKs");
						}

						// ACKs verschicken
						Message ackMsg = new Message(null, addr, "Hallo");
						ackMsg.putHeader(id, hdr.createAck());

						Event ackEvent = new Event(Event.MSG, ackMsg);

						super.down(ackEvent);

						return null;
					} else {
						// msg lokal zwischenspeichern (um auf ACKs zu warten)
						synchronized (locker) {
							msgBuffer.put(hdr.getMsgID(), msg);
						}

						//System.out.println("Nachricht ist da (im msgBuffer)");

						if (leaderFlag) {
							// Ordering Nachricht verschicken
							
							Message orderingMsg = new Message(null, addr, "Hallo");
							VSTotalOrderHeader orderingHdr = VSTotalOrderHeader
									.createMulticast(hdr.getMsgID());
							orderingHdr.setOrderingMessage(true);
							orderingMsg.putHeader(id, orderingHdr);

							Event orderingEvt = new Event(Event.MSG,
									orderingMsg);

							System.out
									.println("ASDKJAL:KJGOIADJGVLK:DJGNLK:ADHGLeader hat OrderingMessage erzeugt und versendet");
							super.down(orderingEvt);
						}
						return null;
					}
				} else {
					// msg lokal zwischenspeichern (um auf ACKs zu warten)
					synchronized (locker) {
						msgBuffer.put(hdr.getMsgID(), msg);
					}

					Message ackMsg = new Message(null, addr, "HALLLLIO");
					ackMsg.putHeader(id, hdr.createAck());

					Event ackEvent = new Event(Event.MSG, ackMsg);

					super.down(ackEvent);
					return null;
				}

			} else if (hdr.getMsgType() == VSTotalOrderMsgType.REROUTING) {
				if (leaderFlag) {
					Message newMsg = new Message(null, addr,
							msg.getRawBuffer(), msg.getOffset(),
							msg.getLength());

					newMsg.putHeader(id,
							VSTotalOrderHeader.createMulticast(hdr.getMsgID()));

					Event modifiedEvent = new Event(Event.MSG, newMsg);
					super.down(modifiedEvent);
					return null;
				}
			} else if (hdr.getMsgType() == VSTotalOrderMsgType.ACK) {

				synchronized (locker) {
					// key existiert noch nicht in msgACKCounter
					if (!msgACKCounter.containsKey(hdr.getMsgID())) {
						msgACKCounter.put(hdr.getMsgID(), 1);
					} else {
						// existiert schon -> inkrementieren um 1
						msgACKCounter.put(hdr.getMsgID(),
								msgACKCounter.get(hdr.getMsgID()) + 1);
					}
				}
				// System.out.println("ACK erhalten: " +
				// msgACKCounter.get(hdr.getMsgID()));

				// wenn alle ACKs da sind -> urspruengliche Nachricht
				// weiterreichen
				if (VSConstants.OPTIMIZED_ACK_STRATEGY) {
					//System.out.println("ACK angekommen........"
							//+ hdr.getMsgID());
					return iterateOrderedMessageQueue();
				} else {
					if (msgACKCounter.get(hdr.getMsgID()) >= view.getMembers()
							.size()) {

						Event modifiedEvent = new Event(Event.MSG,
								msgBuffer.get(hdr.getMsgID()));

						// System.out.println("genug ACKs, Nachricht nach oben");
						return super.up(modifiedEvent);
					} else {
						// System.out.println("noch nicht genug ACKs: "
						// + msgACKCounter.get(hdr.getMsgID()));

						return null;
					}
				}

			} else {
				System.out.println("VSTotalOrderMsgType ist weder "
						+ "MULTICAST noch REROUTING noch ACK");
			}
		}

		case Event.VIEW_CHANGE: {
			View newView = (View) evt.getArg();
			synchronized (locker) {
				view = newView;
				if (!view.getMembers().isEmpty()) {
					leaderAddr = view.getMembers().firstElement();
					leaderFlag = (leaderAddr.equals(addr));
				} else {
					System.err.println("view.getMembers(): leere Liste.");
				}
			}
			return super.up(evt);
		}

		default: {
			return super.up(evt);
		}

		}
	}
}
