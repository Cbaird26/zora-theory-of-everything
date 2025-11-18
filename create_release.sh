#!/bin/bash

# Script to create GitHub Release v1.0.0
# Requires: GITHUB_TOKEN environment variable or GitHub CLI (gh)

REPO="Cbaird26/zora-theory-of-everything"
TAG="v1.0.0"
RELEASE_NAME="🎉 Theory of Everything - COMPLETE (v1.0.0)"
RELEASE_NOTES_FILE="RELEASE_NOTES_v1.0.0.md"

echo "🚀 Creating GitHub Release: $TAG"
echo "Repository: $REPO"
echo ""

# Check if GitHub CLI is available
if command -v gh &> /dev/null; then
    echo "Using GitHub CLI..."
    gh release create "$TAG" \
        --title "$RELEASE_NAME" \
        --notes-file "$RELEASE_NOTES_FILE" \
        --repo "$REPO"
    echo "✅ Release created successfully!"
elif [ -n "$GITHUB_TOKEN" ]; then
    echo "Using GitHub API with token..."
    RELEASE_NOTES=$(cat "$RELEASE_NOTES_FILE" | jq -Rs .)
    
    curl -X POST \
        -H "Authorization: token $GITHUB_TOKEN" \
        -H "Accept: application/vnd.github.v3+json" \
        "https://api.github.com/repos/$REPO/releases" \
        -d "{
            \"tag_name\": \"$TAG\",
            \"name\": \"$RELEASE_NAME\",
            \"body\": $RELEASE_NOTES,
            \"draft\": false,
            \"prerelease\": false
        }"
    echo ""
    echo "✅ Release created successfully!"
else
    echo "⚠️  GitHub CLI (gh) not found and GITHUB_TOKEN not set."
    echo ""
    echo "To create the release, you can:"
    echo "1. Install GitHub CLI: brew install gh (then run: gh auth login)"
    echo "2. Or set GITHUB_TOKEN environment variable and run this script again"
    echo "3. Or manually create release at: https://github.com/$REPO/releases/new"
    echo ""
    echo "Tag to use: $TAG"
    echo "Release notes are in: $RELEASE_NOTES_FILE"
    exit 1
fi

