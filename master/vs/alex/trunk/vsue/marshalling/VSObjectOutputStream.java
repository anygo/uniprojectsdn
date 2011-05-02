package vsue.marshalling;

import java.util.Vector;
import java.lang.reflect.*;
import java.io.OutputStream;
import java.io.IOException;
import java.io.NotSerializableException;
import java.io.Serializable;
import java.nio.*;

public class VSObjectOutputStream {

	private Vector<Object> m_brain;
	private OutputStream m_out;

	public VSObjectOutputStream(OutputStream out) {

		m_brain = new Vector<Object>();
		m_out = out;

	}

	public OutputStream getOutputStream() {

		return m_out;

	}

	public void writeObject(Object object) throws IOException, NotSerializableException {

		/**
		 * 1)   Send control byte
		 *        (byte) 0 : reset m_brain
		 *        (byte) 1 : normal operation
		 *
		 * 2.1) Object is a null-reference
		 *        send int start = 0
		 *
		 * 2.2) Object was already sent
		 *        send int start = -(vector_pos)
		 *
		 * 2.3) Object was not sent before
		 *        send int start = classname.getBytes().length
		 *        send byte[] classname
		 */

		ByteBuffer start = ByteBuffer.allocate(Integer.SIZE/8);

		
		/**
		 * 2.1 Object is a null-reference
		 */
		if(object == null) {
			start.putInt(0);
			m_out.write(1);
			m_out.write(start.array());
			return;
		}

		
		/**
		 * 2.2 Object was already sent
		 */
		int objPos = -1;
		objPos = m_brain.indexOf(object);
		
		if(objPos != -1) {
			int idx = -1 - objPos;
			start.putInt(idx);
			m_out.write(1);
			m_out.write(start.array());
			return;
		}

		
		/**
		 * 2.3) Object was not sent before
		 */
		Class objClass = object.getClass();
		
		// Write control byte
		m_out.write(1);

		// Send classname
		String tmp = objClass.getName();
		byte[] classname = tmp.getBytes("UTF-8");
		start.putInt(classname.length);
		m_out.write(start.array());
		m_out.write(classname);

		// Not serializable check
		if(!Serializable.class.isAssignableFrom(objClass)) {
			throw new NotSerializableException();
		}
		
		// Add object to brain
		m_brain.add(object);
		
		if(objClass.isArray()) {

			int arraylen = Array.getLength(object);
			ByteBuffer lenbuf = ByteBuffer.allocate(Integer.SIZE/8);
			lenbuf.putInt(arraylen);
			m_out.write(lenbuf.array());

			for(int i = 0; i < arraylen; i++) {
				writeObject(Array.get(object, i));
			}

		} else {

			// Don't handle VSExternalizable objects here. Call
			// writeExternal instead
			if(VSExternalizable.class.isAssignableFrom(objClass)) {
				((VSExternalizable)object).writeExternal(this);
				return;
			}

// TODO object superclass handling

			Field[] attributes = objClass.getDeclaredFields();
			for(int i = 0; i < attributes.length; i++) {

				Field attr = attributes[i];
				Class c = attr.getType();
				ByteBuffer buf;
				int m = attr.getModifiers();
				boolean restoreAccess = false;

				// skip if transient or static final
				if(Modifier.isTransient(m) ||
				   (Modifier.isStatic(m) &&
				    Modifier.isFinal(m)) ) {

				     continue;
				}

				if(!attr.isAccessible()) {
					restoreAccess = true;
					attr.setAccessible(true);
				}

				try {
					if(c.isPrimitive()) {

						if(c == boolean.class) {
							buf = ByteBuffer.allocate(1);
							if(attr.getBoolean(object)) {
								buf.put((byte)1);
							} else {
								buf.put((byte)0);
							}
						} else if(c == byte.class) {
							buf = ByteBuffer.allocate(Byte.SIZE/8);
							buf.put(attr.getByte(object));
						} else if(c == char.class) {
							buf = ByteBuffer.allocate(Character.SIZE/8);
							buf.putChar(attr.getChar(object));
						} else if(c == double.class) {
							buf = ByteBuffer.allocate(Double.SIZE/8);
							buf.putDouble(attr.getDouble(object));
						} else if(c == float.class) {
							buf = ByteBuffer.allocate(Float.SIZE/8);
							buf.putFloat(attr.getFloat(object));
						} else if(c == int.class) {
							buf = ByteBuffer.allocate(Integer.SIZE/8);
							buf.putInt(attr.getInt(object));
						} else if(c == long.class) {
							buf = ByteBuffer.allocate(Long.SIZE/8);
							buf.putLong(attr.getLong(object));
						} else { // short.class
							buf = ByteBuffer.allocate(Short.SIZE/8);
							buf.putShort(attr.getShort(object));
						}

						m_out.write(buf.array());

					} else { // -> not primitive
						writeObject(attr.get(object));
					}

				} catch (IllegalAccessException e) {
					e.printStackTrace();
				}

				if(restoreAccess) {
					attr.setAccessible(false);
				}
			}
                }
        }

	public void reset() throws IOException {

		m_brain.clear();
		m_out.write(0);
		m_out.flush();

	}

	public void close() throws IOException {

		reset();
		m_out.close();

	}
}

