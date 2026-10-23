export const maxSpeakers = 16;

export function parseDialogue(script) {
  const turns = [], issues = [];
  let offset = 0, current = null;
  for (const [index, raw] of String(script).split('\n').entries()) {
    const text = raw.replace(/\r$/, '');
    const match = /^([ \t]*)(?:[Ss](\d+)[ \t]*[:：]|<\|speaker:(\d+)\|>)([ \t]*)(.*)$/.exec(text);
    if (match) {
      const speaker = match[3] === undefined ? Number(match[2]) - 1 : Number(match[3]);
      current = {
        speaker, text: match[5], line: index + 1,
        from: offset + match[1].length, to: offset + text.length - match[4].length - match[5].length,
      };
      turns.push(current);
      if (!Number.isInteger(speaker) || speaker < 0 || speaker >= maxSpeakers) {
        issues.push({ kind: 'speaker', line: index + 1 });
      }
    } else if (current) current.text += '\n' + text;
    else if (text.trim()) issues.push({ kind: 'unassigned', line: index + 1 });
    offset += raw.length + 1;
  }
  for (const turn of turns) turn.text = turn.text.trim();
  return { turns, issues };
}

export function formatDialogue(lines, roles) {
  return lines.map(line => {
    const role = roles.find(item => item.id === line.characterId);
    if (!role) return line.text || '';
    const text = String(line.text || '').replace(/^\s*<\|speaker:\d+\|>\s*/, '');
    return `S${role.speaker + 1}\uFF1A${line.direction ? `[${line.direction}] ` : ''}${text}`;
  }).join('\n');
}

export function reconcileDialogue(turns, roles, previous, createId) {
  const entries = turns.flatMap(turn => {
    const role = roles.find(item => item.speaker === turn.speaker);
    return role ? [{ characterId: role.id, text: turn.text }] : [];
  });
  const available = new Set(previous.map((_, index) => index)), matched = new Map();
  const nearest = (entry, index, exact) => {
    let found = -1, distance = Infinity;
    for (const old of available) {
      const candidate = previous[old];
      if (candidate.characterId !== entry.characterId || (exact && candidate.text !== entry.text)) continue;
      if (Math.abs(old - index) < distance) { found = old; distance = Math.abs(old - index); }
    }
    if (found !== -1) { matched.set(index, previous[found]); available.delete(found); }
  };
  // Keep unchanged utterance identities before matching edited utterances by speaker.
  entries.forEach((entry, index) => nearest(entry, index, true));
  entries.forEach((entry, index) => { if (!matched.has(index)) nearest(entry, index, false); });
  return entries.map((entry, index) => ({
    ...matched.get(index), id: matched.get(index)?.id || createId(), ...entry, direction: '',
  }));
}
