// Nupen phone hands: ESP32 = Bluetooth LE mouse for the iPhone (AssistiveTouch), driven by the PC over USB serial.
// Library: HijelHID_BLEMouse (Apache-2.0, github.com/HijelHub/HijelHID_BLEMouse; needs NimBLE-Arduino >= 2.3.8, arduino-esp32 3.x).
// Protocol (115200 baud, one line each way), matching creator/phonetap.py:
//   P            -> OK P <paired 0|1>
//   M dx dy      -> OK      (one relative move, |dx|,|dy| <= 127)
//   D / U        -> OK      (left button down / up, for swipes)
//   C ms         -> OK      (left click held ms = one tap)
//   W dz         -> OK      (scroll)
#include <HijelHID_BLEMouse.h>

HijelBLEMouse mouse("Nupen Hands", "Nupen", 100, 3, false);
String line;

static void reply(const char* s) { Serial.println(s); }

static void handle(String s) {
  s.trim();
  if (s.length() == 0) { reply("ERR empty"); return; }
  char op = s.charAt(0);
  if (op == 'P') { Serial.print("OK P "); Serial.println(mouse.isPaired() ? 1 : 0); return; }
  if (!mouse.isPaired()) { reply("ERR not paired"); return; }
  long a = 0, b = 0;
  int sp = s.indexOf(' ');
  if (sp > 0) {
    String rest = s.substring(sp + 1);
    int sp2 = rest.indexOf(' ');
    a = rest.toInt();
    if (sp2 > 0) b = rest.substring(sp2 + 1).toInt();
  }
  switch (op) {
    case 'M':
      if (a < -127 || a > 127 || b < -127 || b > 127) { reply("ERR range"); return; }
      mouse.move((int8_t)a, (int8_t)b); delay(4); reply("OK"); return;
    case 'D': mouse.press(MouseButton::Left); delay(15); reply("OK"); return;
    case 'U': mouse.release(MouseButton::Left); delay(15); reply("OK"); return;
    case 'C': mouse.click(MouseButton::Left, a > 0 && a < 2000 ? a : 40); reply("OK"); return;
    case 'W': mouse.scroll((int)a); reply("OK"); return;
  }
  reply("ERR unknown");
}

void setup() {
  Serial.begin(115200);
  mouse.begin();
}

void loop() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') { handle(line); line = ""; }
    else if (c != '\r' && line.length() < 64) line += c;
  }
  delay(1);
}
