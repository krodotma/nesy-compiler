import fs from 'fs'
import git from 'isomorphic-git'
import path from 'path'
import crypto from 'crypto'

// Minimal "Amber" Snapshot Tool
// Purpose: Rapidly capture working directory state into a git commit object (dangling)
// Usage: node iso_amber.mjs <repo_dir> <message>

const dir = process.argv[2] || '.'
const message = process.argv[3] || 'Amber Snapshot'

// Reuse exclusion logic logic (simplified for speed)
const SKIP = new Set(['.git', '.pluribus', 'node_modules', '__pycache__', 'dist', 'coverage', 'LOST_FOUND'])

async function run() {
  try {
    // 1. Stage all files (memory-only staging effectively)
    // Actually, iso-git doesn't have "git stash create".
    // We must manually:
    // a. create blobs for all files
    // b. create tree
    // c. create commit

    // For "Amber", we want to capture the *Worktree* state.
    // We will do a 'add .' equivalent but programmatically to verify what we are capturing.
    // But to be FAST (dead man switch), we might just rely on 'git.add' + 'git.commit' to a DETACHED ref?
    // No, we don't want to touch HEAD.

    // Strategy: Create a temporary index, add everything, write tree, write commit.
    
    // However, to avoid messing with the real .git/index, we should maybe just use the real index
    // IF we assume the agent *should* have staged things?
    // No, Amber must save *unstaged* work.

    // FASTEST PATH: Use the standard index.
    // 1. git add . (Update index with current state)
    // 2. git write-tree (Create tree from index)
    // 3. git commit-tree (Create commit object)
    
    // Risk: Modifying the index during a crash might corrupt it?
    // Alternative: We are 10s from death. We need to save.
    
    // Let's iterate. We will use `git.add` to update the index.
    const filePaths = await git.listFiles({ fs, dir })
    
    // Filter skippables just in case listFiles includes them (it respects .gitignore usually)
    // But we also want to catch *untracked* files that might be ignored but important? 
    // No, standard gitignore rules apply.

    // Add all known files + new files.
    // git.statusMatrix is expensive.
    // git.add({ dir, filepath: '.' }) is recursive.
    
    await git.add({ fs, dir, filepath: '.' })
    
    // Create the snapshot commit
    const sha = await git.commit({
      fs,
      dir,
      message: `[AMBER] ${message}`,
      author: {
        name: 'Amber Preservation',
        email: 'amber@pluribus.os'
      },
      ref: undefined, // Do NOT update HEAD. This creates a dangling commit.
      noUpdateBranch: true // Explicitly do not update any branch.
    })

    console.log(sha)
    process.exit(0)

  } catch (err) {
    console.error('AMBER_FAILURE:', err.message)
    process.exit(1)
  }
}

run()
