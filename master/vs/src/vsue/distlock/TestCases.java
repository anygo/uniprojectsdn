package vsue.distlock;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicBoolean;

import org.jgroups.Address;
import org.jgroups.ChannelClosedException;
import org.jgroups.ChannelNotConnectedException;
import org.jgroups.JChannel;
import org.jgroups.Message;
import org.jgroups.ReceiverAdapter;
import org.jgroups.View;
import org.jgroups.conf.ClassConfigurator;

public final class TestCases extends ReceiverAdapter {
	private static int num_members = 0;
	private static final String CLUSTER_NAME = "gruppe999";
	private static VSLamportLock lamport_lock;
	private static AtomicBoolean runnable = new AtomicBoolean(false);
	private static AtomicBoolean answered = new AtomicBoolean(false);
	private static View view;
	private static TestCases instance;
	private static int value;
	private static JChannel group_comm;
	private static int result_buf;
	
	@Override
	public void receive(Message msg) {
		if (! runnable.get()) return;
		
		DataInputStream dis = new DataInputStream(new ByteArrayInputStream(
			msg.getBuffer()));
		
		try {
			Message reply;
			int otherval;
			byte msgtype = dis.readByte();

			switch (msgtype) {
			case 0: // Simple get
				//System.out.println("GET from " + msg.getSrc().toString());
			
				ByteArrayOutputStream bos = new ByteArrayOutputStream();
				DataOutputStream dos = new DataOutputStream(bos);
				reply = msg.makeReply();
				dos.writeByte(1);
				synchronized (instance) {
					dos.writeInt(value);
				}
				dos.flush();
				reply.setBuffer(bos.toByteArray());
				group_comm.send(reply);
				break;
				
			case 1: // Get reply handling
				//System.out.println("ANSWER from " + msg.getSrc().toString());
				synchronized(answered) {
					answered.set(true);
					answered.notify();
					result_buf = dis.readInt();
				}
				break;
				
			case 2: // Simple set
				//System.out.println("SET from " + msg.getSrc().toString());
				otherval = dis.readInt();
				synchronized (instance) {
					value = otherval;
				}
				reply = msg.makeReply();
				reply.setBuffer(new byte[] { 1, 0, 0, 0, 0 });
				group_comm.send(reply);
				break;
			}
		} catch (IOException e) {
			System.out.println("Invalid I/O: " + e.getMessage());
			e.printStackTrace();
		} catch (ChannelClosedException e) {
		} catch (ChannelNotConnectedException e) {
		}
	}
	
	// Shutdown with some delay
	private static class Killer extends Thread {
		@Override
		public void run() {
			System.err.println(
				"--> View change detected, shutting down in 3 seconds... <--");
			try {sleep(3000);} catch (Exception e){}
			System.exit(0);
		}
	}
	
	@Override
	// Helper method to start the main process when all members are available
	public void viewAccepted(View v) {
		// Shutdown if we get a view change during runtime
		// The lock protocol we're testing won't be able to cope with that...
		if (runnable.get()) {
			group_comm.close();
			new Killer().start();
		}
		
		if (v.size() == num_members) {
			synchronized(runnable) {
				runnable.set(true);
				view = v;
				runnable.notify();
			}
		}
	}
		
	private static void sleepRandomly() {
		int duration = (int)(Math.random() * 2 + 0);
		try { Thread.sleep(duration); } catch(Exception e){}
	}
	
	private static void simpleProtocolTest() throws Exception {
		// Run the simple protocol test
		while (true) {

			System.out.print("?");
			System.out.flush();
			lamport_lock.lock();
			
			sleepRandomly();
			
			System.out.print(".");
			System.out.flush();
			sleepRandomly();
			lamport_lock.unlock();
		}
	}
	
	private static int queryMember(Address member) throws Exception {
		int retval;
		Message m = new Message(member);
		m.setBuffer(new byte[] {0});
		group_comm.send(m);
		synchronized(answered) {
			while (! answered.get()) {
				answered.wait();
			}
			answered.set(false);
			retval = result_buf;
		}
		return retval;
	}
	private static void	setMemberValue(Address member, int value) throws Exception {
		Message m = new Message(member);
		ByteArrayOutputStream bos = new ByteArrayOutputStream();
		DataOutputStream dos = new DataOutputStream(bos);
		dos.writeByte(2);
		dos.writeInt(value);
		dos.flush();
		m.setBuffer(bos.toByteArray());
		group_comm.send(m);
		synchronized(answered) {
			while (! answered.get()) {
				answered.wait();
			}
			answered.set(false);
		}
	}
	
	private static void fancyProtocolTest() throws Exception {
		int iterationcount = 0;
		
		// Run the fancy protocol test
		while (true) {
			sleepRandomly();
			lamport_lock.lock();
			
			if (iterationcount % 100 == 0) {
				int sum = 0;
				for (int i = 0; i < num_members; i++) {
					sum += queryMember(view.getMembers().get(i));
				}
				System.out.println("Sum is " + sum);
			} else {
				int other_id = (int)(Math.random() * view.size());
				Address other_addr = view.getMembers().get(other_id);
				int remote_val = queryMember(other_addr);
				int mov = (int)(Math.random() * remote_val);
				remote_val -= mov;
				System.out.println("Stealing " + mov + " credits from " 
					+ other_addr.toString());
				setMemberValue(other_addr, remote_val);
				synchronized (instance) {
					value += mov;
				}
			}
			
			lamport_lock.unlock();
			iterationcount++;
		}
	}
	
	public static void main(String[] args) throws Exception {
		boolean fancy = false;
		if (args.length < 1) {
			System.err.println("Invalid command line.");
			System.exit(1);
		}
		
		if (args.length > 1 && args[1].equals("fancy"))
			fancy = true;
		
		num_members = Integer.parseInt(args[0]); 
		if (num_members < 2 || num_members > 10) {
			System.err.println("Invalid command line.");
			System.exit(1);
		}
						
		// IPv6 is sometimes broken with JGroups, disable it...
		System.setProperty("java.net.preferIPv4Stack", "true");
		// Override configuration for Java Logging facilities
		System.setProperty("java.util.logging.config.file", "logging.cfg");
		
		ClassConfigurator.addProtocol((short)1800, FIFOUnicast.class);
		ClassConfigurator.addProtocol((short)2000, VSLamportLockProtocol.class);
		ClassConfigurator.addProtocol((short)2001, VSLogicalClockProtocol.class);
		
		// Start group communication and initialize global lock
		group_comm = new JChannel("stack.xml");
		//group_comm.getProtocolStack().addProtocol(new SEQUENCER());
		System.err.println(group_comm.getProtocolStack().printProtocolSpec(false));
		instance = new TestCases();
		group_comm.setReceiver(instance);
		group_comm.connect(CLUSTER_NAME);
		lamport_lock = new VSLamportLock(group_comm);

		value = (int)(Math.random() * 1000);
		// Wait until all communication partners show up in the view
		synchronized(runnable) {
			while (! runnable.get()) {
				runnable.wait();
			}
		}
		
		if (fancy)
			fancyProtocolTest();
		else
			simpleProtocolTest();
	}
}
