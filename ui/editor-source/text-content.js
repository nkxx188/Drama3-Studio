// Keep this spoken-text measure aligned with drama_low_vram.text_chunks.
export function contentWidth(value) {
  const text = String(value ?? ''), bracketEnds = new Map(), stack = [];
  for (let index = 0; index < text.length; index++) {
    if (text[index] === '[') stack.push(index);
    else if (text[index] === ']' && stack.length) bracketEnds.set(stack.pop(), index + 1);
  }
  const phoneme = /<\|phoneme_start\|>([^<>]*?)<\|phoneme_end\|>/y;
  const angle = /<[^<>\r\n]*>/y;
  let width = 0, offset = 0, speakerSpace = false;
  while (offset < text.length) {
    if (speakerSpace) {
      const isSeparator = text[offset] === ' ';
      speakerSpace = false;
      if (isSeparator) { offset++; continue; }
    }
    if (bracketEnds.has(offset)) { offset = bracketEnds.get(offset); continue; }
    if (text[offset] === '<') {
      phoneme.lastIndex = offset;
      const pronunciation = phoneme.exec(text);
      if (pronunciation) {
        const syllables = pronunciation[1].trim().split(/\s+/).filter(Boolean);
        width += syllables.length && syllables.every(part => /^[a-z\u00fcv\u00ea]+[1-5]$/i.test(part))
          ? syllables.length : Array.from(pronunciation[1]).length;
        offset = phoneme.lastIndex; continue;
      }
      angle.lastIndex = offset;
      const control = angle.exec(text);
      if (control) {
        speakerSpace = /^<\|speaker:\d+\|>$/.test(control[0]);
        offset = angle.lastIndex; continue;
      }
    }
    width++;
    offset += text.codePointAt(offset) > 0xffff ? 2 : 1;
  }
  return width;
}
