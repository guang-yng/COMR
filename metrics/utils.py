import re
import multiprocessing
from tqdm import tqdm

def _remove_comments_in_voice(voice_lines, is_truncated=False):
    voice_line = ''
    lyrics_lines = []
    lyrics_id = None
    for idx, line in enumerate(voice_lines[1:]):
        m = re.search(r'%\d+$', line)
        if m:
            voice_line += line[:m.start()]
            lyrics_id = 0
        elif line.startswith('w:'):
            if lyrics_id is None:
                return None
            if len(lyrics_lines) <= lyrics_id:
                lyrics_lines.append('w:')
            lyrics_lines[lyrics_id] += line[2:]
            lyrics_id += 1
        elif is_truncated and (idx == len(voice_line) - 1): # Handle truncated case
            voice_line += line
        else:
            return None
    return [voice_lines[0], voice_line, *lyrics_lines]
    

def remove_comments(transcription):
    """
    Remove comments and merge each voice into a single line.
    Return None if transcription is not well-formed.
    """
    lines = transcription.split('\n')
    if lines[-1] != '':
        lines.append('')
    prev_voice_header = None
    new_lines = []
    voices_found = []
    for idx, line in enumerate(lines):
        m = re.match(r'V:(\d+)', line)
        if idx == len(lines)-1 or (m and m.group(1) in voices_found):
            if prev_voice_header is not None:
                # print(prev_voice_header, idx)
                new_voice_lines = _remove_comments_in_voice(lines[prev_voice_header:idx], idx==len(lines)-1)
                if new_voice_lines is None:
                    return None
                new_lines.extend(new_voice_lines)
            else:
                new_lines.extend(lines[:idx])
            prev_voice_header = idx
        elif m:
            voices_found.append(m.group(1))

    return '\n'.join(new_lines) + ('\n' if transcription.endswith('\n') else '')


def remove_comments_batch(transcriptions, num_workers=8):
    with multiprocessing.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(remove_comments, transcriptions),
            desc=f"removing comments and merging lines...",
            total=len(transcriptions)
        ))
    return results