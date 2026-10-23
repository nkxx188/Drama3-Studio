// English prompt values follow Fish Audio's emotion reference; captions are localized by I18n.
export const tagCategories = [
  ['emotions', 'Emotions'], ['tone', 'Tone'], ['sounds', 'Human sounds'],
  ['effects', 'Effects & pauses'], ['other', 'Other delivery'],
];

export const tagSections = [
  { category: 'emotions', title: 'Basic emotions', tags: [
    ['happy', 'Happy'], ['sad', 'Sad'], ['angry', 'Angry'], ['excited', 'Excited'],
    ['calm', 'Calm'], ['nervous', 'Nervous'], ['confident', 'Confident'], ['surprised', 'Surprised'],
    ['satisfied', 'Satisfied'], ['delighted', 'Delighted'], ['scared', 'Scared'], ['worried', 'Worried'],
    ['upset', 'Upset'], ['frustrated', 'Frustrated'], ['depressed', 'Depressed'], ['empathetic', 'Empathetic'],
    ['embarrassed', 'Embarrassed'], ['disgusted', 'Disgusted'], ['moved', 'Moved'], ['proud', 'Proud'],
    ['relaxed', 'Relaxed'], ['grateful', 'Grateful'], ['curious', 'Curious'], ['sarcastic', 'Sarcastic'],
  ] },
  { category: 'emotions', title: 'Advanced emotions', tags: [
    ['disdainful', 'Disdainful'], ['unhappy', 'Unhappy'], ['anxious', 'Anxious'], ['hysterical', 'Hysterical'],
    ['indifferent', 'Indifferent'], ['uncertain', 'Uncertain'], ['doubtful', 'Doubtful'], ['confused', 'Confused'],
    ['disappointed', 'Disappointed'], ['regretful', 'Regretful'], ['guilty', 'Guilty'], ['ashamed', 'Ashamed'],
    ['jealous', 'Jealous'], ['envious', 'Envious'], ['hopeful', 'Hopeful'], ['optimistic', 'Optimistic'],
    ['pessimistic', 'Pessimistic'], ['nostalgic', 'Nostalgic'], ['lonely', 'Lonely'], ['bored', 'Bored'],
    ['contemptuous', 'Contemptuous'], ['sympathetic', 'Sympathetic'], ['compassionate', 'Compassionate'],
    ['determined', 'Determined'], ['resigned', 'Resigned'],
  ] },
  { category: 'tone', title: 'Tone markers', tags: [
    ['in a hurry tone', 'Hurried tone'], ['shouting', 'Shouting'], ['screaming', 'Screaming'],
    ['whispering', 'Whispering'], ['soft tone', 'Soft tone'], ['emphasis', 'Emphasis'],
  ] },
  { category: 'sounds', title: 'Human sounds', tags: [
    ['laughing', 'Laughing', ['laughs']], ['chuckling', 'Chuckling', ['chuckles']],
    ['sobbing', 'Sobbing', ['sobs']], ['crying loudly', 'Crying loudly'],
    ['sighing', 'Sighing', ['sighs']], ['groaning', 'Groaning'], ['panting', 'Panting'],
    ['gasping', 'Gasping', ['gasps']], ['yawning', 'Yawning'], ['snoring', 'Snoring'],
    ['clear throat', 'Clear throat', ['clears throat']],
  ] },
  { category: 'effects', title: 'Effects & pauses', tags: [
    ['audience laughing', 'Audience laughter'], ['background laughter', 'Background laughter'],
    ['crowd laughing', 'Crowd laughter'], ['break', 'Short pause'], ['long-break', 'Long pause'],
  ] },
  { category: 'other', title: 'Other delivery', tags: [
    ['breathy', 'Breathy'], ['warm', 'Warm'], ['pause', 'Pause'],
  ] },
];
