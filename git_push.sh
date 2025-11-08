#!/bin/bash

# Git 자동 푸시 스크립트
# 사용법: ./git_push.sh "커밋 메시지"

# 커밋 메시지 확인
if [ -z "$1" ]; then
    echo "❌ 커밋 메시지를 입력해주세요!"
    echo "사용법: ./git_push.sh \"커밋 메시지\""
    exit 1
fi

COMMIT_MSG="$1"

echo "🔍 Git 상태 확인 중..."
git status

echo ""
echo "📦 변경사항 추가 중..."
git add .

echo ""
echo "💾 커밋 생성 중..."
git commit -m "$COMMIT_MSG"

if [ $? -eq 0 ]; then
    echo ""
    echo "🚀 GitHub에 푸시 중..."
    git push
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ GitHub 푸시 완료!"
        echo "🔗 https://github.com/YuMinBee/RLLMDRIVE"
    else
        echo ""
        echo "❌ 푸시 실패!"
    fi
else
    echo ""
    echo "⚠️  커밋할 변경사항이 없습니다."
fi
