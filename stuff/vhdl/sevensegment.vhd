-- vim: set syntax=vhdl ts=8 sts=2 sw=2 et:

---------------------------------------------------------------------
-- Project:     Digitaler Wecker
-- File:        Segment Decoder
-- Language:    VHDL
-- Modifications:
---------------------------------------------------------------------
LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.STD_LOGIC_ARITH.ALL;
USE IEEE.STD_LOGIC_UNSIGNED.ALL;

ENTITY sevensegment IS
  PORT (NUMBER: IN integer RANGE 0 TO 9;
        DIGIT: OUT STD_LOGIC_VECTOR (6 DOWNTO 0) );
END sevensegment;

ARCHITECTURE Behavioral OF sevensegment IS
  SIGNAL DIGIT_in : STD_LOGIC_VECTOR (6 DOWNTO 0);
BEGIN
  PROCESS(NUMBER) 
  BEGIN 
    -- |-0-|
    -- 6   1
    -- |-5-|
    -- 4   2
    -- |-3-|
    CASE NUMBER IS -- Reihenfolge 6543210
      WHEN 0      => DIGIT_in <= "1011111"
      WHEN 1      => DIGIT_in <= "0000110"
      WHEN 2      => DIGIT_in <= "0111011"
      WHEN 3      => DIGIT_in <= "0101111"
      WHEN 4      => DIGIT_in <= "1100110"
      WHEN 5      => DIGIT_in <= "1111101"
      WHEN 6      => DIGIT_in <= "0000111"
      WHEN 7      => DIGIT_in <= "0000111"
      WHEN 8      => DIGIT_in <= "1111111"
      WHEN 9      => DIGIT_in <= "1101111"
      WHEN others => DIGIT_in <= "1110001"    -- "F"
    END CASE; 
  END PROCESS;
  DIGIT <= not DIGIT_in;
END Behavioral;
