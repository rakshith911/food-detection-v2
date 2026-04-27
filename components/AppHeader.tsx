import React from 'react';
import { StyleSheet, View, Text, StyleProp, ViewStyle } from 'react-native';
import ProfileIcon from './ProfileIcon';

interface AppHeaderProps {
  displayName: string;
  lastLoginDate?: string;
  lastLoginTime?: string;
  onProfilePress?: () => void;
  profileIconColor?: string;
  containerStyle?: StyleProp<ViewStyle>;
}

export default function AppHeader({
  displayName,
  lastLoginDate,
  lastLoginTime,
  onProfilePress,
  profileIconColor = '#3B82F6',
  containerStyle,
}: AppHeaderProps) {
  return (
    <View style={[styles.header, containerStyle]}>
      <View style={styles.row}>
        <ProfileIcon
          size={24}
          color={profileIconColor}
          onPress={onProfilePress}
          style={styles.avatar}
        />
        <View style={styles.textContainer}>
          <Text style={styles.greeting} numberOfLines={1}>
            Hello {displayName}!
          </Text>
          {lastLoginDate ? (
            <Text style={styles.subText} numberOfLines={1}>
              Last Login: {lastLoginTime ? `${lastLoginDate}, ${lastLoginTime}` : lastLoginDate}
            </Text>
          ) : null}
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  header: {
    backgroundColor: '#FFFFFF',
    borderBottomWidth: 1,
    borderBottomColor: '#E5E7EB',
    paddingHorizontal: 16,
    paddingVertical: 18,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  avatar: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: '#E0F2FE',
    borderWidth: 1,
    borderColor: '#7BA21B',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
    overflow: 'hidden',
  },
  textContainer: {
    flex: 1,
  },
  greeting: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1F2937',
    marginBottom: 2,
  },
  subText: {
    fontSize: 11,
    color: '#6B7280',
  },
});
